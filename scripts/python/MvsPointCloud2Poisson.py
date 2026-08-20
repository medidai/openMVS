#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
Reconstruct a mesh from a point cloud using Poisson Surface Reconstruction.
Supports automatic normal estimation, outlier filtering, and adaptive parameter estimation.

Install:
  pip install open3d numpy tqdm argparse

Example usage:
  python3 MvsPointCloud2Poisson.py -i input_cloud.ply -o output_mesh.ply
  python3 MvsPointCloud2Poisson.py -i input.ply -o output.ply --depth 0 --density_threshold 0.01
  python3 MvsPointCloud2Poisson.py -i input.ply -o output.ply --filter_outliers statistical --filter_nb_neighbors 20
"""

import numpy as np
import argparse
import os
import open3d as o3d
from tqdm import tqdm


def validate_and_clean_points(pcd, verbose=True):
    """
    Validate points for finite values and remove invalid points.

    Args:
        pcd: Open3D PointCloud object
        verbose: Print statistics

    Returns:
        Cleaned PointCloud, number of invalid points removed
    """
    points = np.asarray(pcd.points)
    has_normals = pcd.has_normals()
    has_colors = pcd.has_colors()

    # Check for finite values in points
    valid_mask = np.all(np.isfinite(points), axis=1)

    # Check normals if present
    if has_normals:
        normals = np.asarray(pcd.normals)
        valid_normals_mask = np.all(np.isfinite(normals), axis=1)
        valid_mask = valid_mask & valid_normals_mask

    num_invalid = np.sum(~valid_mask)

    if num_invalid > 0:
        if verbose:
            print(f"Warning: Removed {num_invalid} invalid points with non-finite coordinates/normals.")

        # Create new point cloud with valid points only
        pcd_clean = o3d.geometry.PointCloud()
        pcd_clean.points = o3d.utility.Vector3dVector(points[valid_mask])

        if has_normals:
            pcd_clean.normals = o3d.utility.Vector3dVector(normals[valid_mask])

        if has_colors:
            colors = np.asarray(pcd.colors)
            pcd_clean.colors = o3d.utility.Vector3dVector(colors[valid_mask])

        return pcd_clean, num_invalid

    return pcd, 0


def normalize_point_cloud(pcd, target_radius=100.0):
    """
    Center point cloud at origin and normalize to target radius.

    Args:
        pcd: Open3D PointCloud object
        target_radius: Target radius for normalization

    Returns:
        Normalized PointCloud, center offset, scale factor
    """
    points = np.asarray(pcd.points)

    # Compute center
    center = np.mean(points, axis=0)

    # Center the point cloud
    points_centered = points - center

    # Compute scale (max distance from center)
    distances = np.linalg.norm(points_centered, axis=1)
    max_dist = np.max(distances)

    # Scale to target radius
    scale = target_radius / max_dist if max_dist > 0 else 1.0
    points_normalized = points_centered * scale

    # Create normalized point cloud
    pcd_normalized = o3d.geometry.PointCloud()
    pcd_normalized.points = o3d.utility.Vector3dVector(points_normalized)

    if pcd.has_normals():
        # Normals are directions, no translation/scaling needed, just copy
        pcd_normalized.normals = pcd.normals

    if pcd.has_colors():
        pcd_normalized.colors = pcd.colors

    return pcd_normalized, center, scale


def denormalize_mesh(mesh, center, scale):
    """
    Transform mesh back to original coordinate system.

    Args:
        mesh: Open3D TriangleMesh object
        center: Original center offset
        scale: Scale factor used for normalization

    Returns:
        Denormalized mesh
    """
    vertices = np.asarray(mesh.vertices)

    # Reverse normalization: scale back and translate
    vertices_denormalized = (vertices / scale) + center

    mesh.vertices = o3d.utility.Vector3dVector(vertices_denormalized)

    return mesh


def estimate_normals(pcd, search_radius=None, max_nn=30):
    """
    Estimate normals for point cloud if they don't exist.

    Args:
        pcd: Open3D PointCloud object
        search_radius: Search radius for normal estimation (auto-estimated if None)
        max_nn: Maximum number of nearest neighbors

    Returns:
        PointCloud with normals
    """
    points = np.asarray(pcd.points)

    if search_radius is None:
        # Auto-estimate search radius based on point cloud density
        # Use a small sample for efficiency
        num_samples = min(3000, len(points))
        sample_indices = np.random.choice(len(points), num_samples, replace=False)
        sample = points[sample_indices]

        # Build KDTree and compute average nearest neighbor distance
        pcd_sample = o3d.geometry.PointCloud()
        pcd_sample.points = o3d.utility.Vector3dVector(sample)
        kdtree = o3d.geometry.KDTreeFlann(pcd_sample)

        nn_distances = []
        for i in range(min(1000, len(sample))):
            [_, idx, dist] = kdtree.search_knn_vector_3d(sample[i], 2)  # k=2 to get nearest neighbor
            if len(dist) > 1:
                nn_distances.append(np.sqrt(dist[1]))

        avg_spacing = np.median(nn_distances) if nn_distances else 1.0
        search_radius = avg_spacing * 3.0  # Use 3x average spacing

    print(f"Estimating normals with search radius: {search_radius:.4f}")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_nn)
    )

    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=max_nn)

    return pcd


def normalize_normals(pcd):
    """
    Normalize all normal vectors to unit length.

    Args:
        pcd: Open3D PointCloud object with normals

    Returns:
        PointCloud with normalized normals
    """
    if not pcd.has_normals():
        return pcd

    normals = np.asarray(pcd.normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)

    # Avoid division by zero
    norms = np.maximum(norms, 1e-8)

    normals_normalized = normals / norms
    pcd.normals = o3d.utility.Vector3dVector(normals_normalized)

    return pcd


def filter_outliers(pcd, method='statistical', nb_neighbors=20, std_ratio=2.0,
                   radius=None, nb_points=16, verbose=True):
    """
    Filter outlier points from point cloud.

    Args:
        pcd: Open3D PointCloud object
        method: 'statistical' or 'radius'
        nb_neighbors: Number of neighbors for statistical outlier removal
        std_ratio: Standard deviation ratio for statistical outlier removal
        radius: Radius for radius outlier removal (auto-estimated if None)
        nb_points: Minimum number of points in radius
        verbose: Print statistics

    Returns:
        Filtered PointCloud, number of outliers removed
    """
    original_count = len(pcd.points)

    if method == 'statistical':
        if verbose:
            print(f"Filtering outliers (statistical: nb_neighbors={nb_neighbors}, std_ratio={std_ratio})...")
        pcd_filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                            std_ratio=std_ratio)
    elif method == 'radius':
        if radius is None:
            # Auto-estimate radius from point cloud
            points = np.asarray(pcd.points)
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox_diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
            radius = bbox_diag * 0.01  # 1% of bounding box diagonal

        if verbose:
            print(f"Filtering outliers (radius: radius={radius:.4f}, nb_points={nb_points})...")
        pcd_filtered, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    else:
        if verbose:
            print("No outlier filtering applied.")
        return pcd, 0

    num_removed = original_count - len(pcd_filtered.points)
    if verbose:
        print(f"Removed {num_removed} outlier points ({num_removed/original_count*100:.2f}%)")

    return pcd_filtered, num_removed


def estimate_poisson_depth(pcd, num_samples=3000):
    """
    Estimate appropriate Poisson reconstruction depth based on point cloud density.

    Args:
        pcd: Open3D PointCloud object
        num_samples: Number of samples for density estimation

    Returns:
        Estimated octree depth
    """
    points = np.asarray(pcd.points)

    # Compute bounding box diagonal
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())

    # Estimate average spacing using nearest neighbor distances
    num_samples = min(num_samples, len(points))
    sample_indices = np.random.choice(len(points), num_samples, replace=False)
    sample = points[sample_indices]

    pcd_sample = o3d.geometry.PointCloud()
    pcd_sample.points = o3d.utility.Vector3dVector(sample)
    kdtree = o3d.geometry.KDTreeFlann(pcd_sample)

    nn_distances = []
    for i in range(min(1000, len(sample))):
        [_, idx, dist] = kdtree.search_knn_vector_3d(sample[i], 2)  # k=2 to get nearest neighbor
        if len(dist) > 1:
            nn_distances.append(np.sqrt(dist[1]))

    if not nn_distances:
        return 8  # Default fallback

    avg_spacing = np.median(nn_distances)

    # Estimate depth: depth ~ log2(bbox_diagonal / avg_spacing) + offset
    # The offset ensures we have enough resolution
    estimated_depth = int(np.log2(bbox_diag / avg_spacing)) + 2

    # Clamp to reasonable range
    estimated_depth = max(5, min(estimated_depth, 12))

    return estimated_depth


def remove_low_density_vertices(mesh, densities, threshold, verbose=True):
    """
    Remove vertices with density below threshold.

    Args:
        mesh: Open3D TriangleMesh object
        densities: Vertex density values from Poisson reconstruction
        threshold: Minimum density threshold (vertices below this are removed)
        verbose: Print statistics

    Returns:
        Filtered mesh
    """
    if threshold <= 0:
        return mesh

    densities_array = np.asarray(densities)
    vertices_to_remove = densities_array < threshold

    num_removed = np.sum(vertices_to_remove)
    total_vertices = len(mesh.vertices)

    if verbose:
        print(f"Density statistics:")
        print(f"  Min: {np.min(densities_array):.6f}")
        print(f"  Max: {np.max(densities_array):.6f}")
        print(f"  Mean: {np.mean(densities_array):.6f}")
        print(f"  Median: {np.median(densities_array):.6f}")
        print(f"  Threshold: {threshold:.6f}")
        print(f"Removing {num_removed} vertices ({num_removed/total_vertices*100:.2f}%) with density < {threshold:.6f}")

    # Remove vertices below threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct a mesh from a point cloud using Poisson Surface Reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    parser.add_argument("-i", "--input", type=str, required=True,
                       help="Path to input point cloud (PLY, PCD, XYZ)")
    parser.add_argument("-o", "--output", type=str, default="mesh.ply",
                       help="Path to output mesh file (PLY, OBJ)")

    # Normalization
    parser.add_argument("--normalize_radius", type=float, default=100.0,
                       help="Target radius for point cloud normalization")

    # Poisson parameters
    parser.add_argument("--depth", type=int, default=0,
                       help="Octree depth for Poisson reconstruction (0 for auto-estimation)")
    parser.add_argument("--width", type=float, default=0.0,
                       help="Target width of finest level octree cells (0 for depth-based)")
    parser.add_argument("--scale", type=float, default=1.1,
                       help="Ratio between reconstruction cube and bounding cube")
    parser.add_argument("--linear_fit", action='store_true',
                       help="Use linear interpolation for iso-surface extraction")

    # Density filtering
    parser.add_argument("--density_threshold", type=float, default=0.0,
                       help="Remove vertices with density below this threshold (0 to disable)")

    # Outlier filtering
    parser.add_argument("--filter_outliers", type=str, choices=['none', 'statistical', 'radius'],
                       default='none',
                       help="Outlier filtering method")
    parser.add_argument("--filter_nb_neighbors", type=int, default=20,
                       help="Number of neighbors for statistical outlier removal")
    parser.add_argument("--filter_std_ratio", type=float, default=2.0,
                       help="Standard deviation ratio for statistical outlier removal")
    parser.add_argument("--filter_radius", type=float, default=0.0,
                       help="Radius for radius outlier removal (0 for auto-estimation)")
    parser.add_argument("--filter_nb_points", type=int, default=16,
                       help="Minimum number of points in radius for radius outlier removal")

    # Normal estimation
    parser.add_argument("--normal_search_radius", type=float, default=0.0,
                       help="Search radius for normal estimation (0 for auto-estimation)")
    parser.add_argument("--normal_max_nn", type=int, default=30,
                       help="Maximum nearest neighbors for normal estimation")

    # Verbose
    parser.add_argument("-v", "--verbose", action='store_true',
                       help="Print detailed progress information")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Load point cloud
    print(f"Loading {args.input}...")
    try:
        pcd = o3d.io.read_point_cloud(args.input)
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return

    if len(pcd.points) == 0:
        print("Error: Point cloud is empty.")
        return

    print(f"Loaded {len(pcd.points)} points.")

    # Check for colors
    has_colors = pcd.has_colors()
    if args.verbose and has_colors:
        print("Point cloud has color information.")

    # Validate and clean points
    pcd, num_invalid = validate_and_clean_points(pcd, verbose=True)

    if len(pcd.points) == 0:
        print("Error: No valid points remaining after cleaning.")
        return

    # Normalize point cloud
    print(f"Normalizing point cloud to origin and radius {args.normalize_radius}...")
    pcd_normalized, original_center, original_scale = normalize_point_cloud(pcd, args.normalize_radius)
    if args.verbose:
        print(f"  Original center: [{original_center[0]:.4f}, {original_center[1]:.4f}, {original_center[2]:.4f}]")
        print(f"  Scale factor: {original_scale:.4f}")

    # Check for normals
    has_normals = pcd_normalized.has_normals()

    if not has_normals:
        print("Warning: Point cloud does not have normals. Estimating normals...")
        print("  Note: Reconstructed mesh may be less accurate without original normals.")
        search_radius = args.normal_search_radius if args.normal_search_radius > 0 else None
        pcd_normalized = estimate_normals(pcd_normalized, search_radius, args.normal_max_nn)
    else:
        print("Point cloud has normals. Normalizing them...")
        pcd_normalized = normalize_normals(pcd_normalized)

    # Filter outliers
    if args.filter_outliers != 'none':
        pcd_normalized, num_outliers = filter_outliers(
            pcd_normalized,
            method=args.filter_outliers,
            nb_neighbors=args.filter_nb_neighbors,
            std_ratio=args.filter_std_ratio,
            radius=args.filter_radius if args.filter_radius > 0 else None,
            nb_points=args.filter_nb_points,
            verbose=True
        )

        if len(pcd_normalized.points) == 0:
            print("Error: No points remaining after outlier filtering.")
            return

    # Determine Poisson depth
    if args.depth == 0:
        poisson_depth = estimate_poisson_depth(pcd_normalized)
        print(f"Auto-estimated Poisson depth: {poisson_depth}")
    else:
        poisson_depth = args.depth
        print(f"Using specified Poisson depth: {poisson_depth}")

    # Run Poisson reconstruction
    print("Running Poisson surface reconstruction...")
    with tqdm(total=100, desc="Poisson reconstruction") as pbar:
        if args.width > 0:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd_normalized,
                depth=poisson_depth,
                width=args.width,
                scale=args.scale,
                linear_fit=args.linear_fit
            )
        else:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd_normalized,
                depth=poisson_depth,
                scale=args.scale,
                linear_fit=args.linear_fit
            )
        pbar.update(100)

    print(f"Generated mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")

    # Remove low-density vertices
    if args.density_threshold > 0:
        mesh = remove_low_density_vertices(mesh, densities, args.density_threshold, verbose=True)
        print(f"Mesh after density filtering: {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")

    # Denormalize mesh back to original coordinate system
    print("Transforming mesh back to original coordinate system...")
    mesh = denormalize_mesh(mesh, original_center, original_scale)

    # Compute vertex normals if not present
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Save mesh
    print(f"Saving mesh to {args.output}...")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        success = o3d.io.write_triangle_mesh(args.output, mesh)
        if success:
            print("Done!")
        else:
            print("Error: Failed to save mesh.")
    except Exception as e:
        print(f"Error saving mesh: {e}")


if __name__ == "__main__":
    main()
