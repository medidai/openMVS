#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
Reconstruct a mesh from a fused point cloud with normals using TSDF and Marching Cubes.
Uses a vectorized splatting approach for efficient SDF generation.

Install:
  pip install numpy scikit-image plyfile tqdm argparse

Example usage:
  python3 MvsPointCloud2TSDF.py -i input_cloud.ply [-o output_mesh.ply] [--voxel_size VOXEL_SIZE] [--truncation_mult TRUNCATION_MULT]
"""

import numpy as np
import argparse
import os
from plyfile import PlyData, PlyElement
from skimage.measure import marching_cubes
from tqdm import tqdm
from scipy.spatial import cKDTree

def load_ply(path):
    """
    Load points and normals from a PLY file.
    """
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    
    points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    
    if 'nx' in vertex.data.dtype.names and 'ny' in vertex.data.dtype.names and 'nz' in vertex.data.dtype.names:
        normals = np.stack([vertex['nx'], vertex['ny'], vertex['nz']], axis=-1)
    else:
        raise ValueError("PLY file must contain normals (nx, ny, nz).")
        
    return points, normals

def estimate_voxel_size(points, num_samples=3000):
    """
    Estimate voxel size based on nearest neighbor distances of a subset of points.
    Using a brute-force approach on a small sample to avoid scipy dependency if possible,
    but simplest is just to take 1% of bounding box diagonal or similar heuristic
    if we want to avoid KDTree/scipy completely. 
    However, decent estimation requires spatial awareness.
    
    Let's use a simple heuristic for now: 
    Average distance to nearest neighbor in a small random subset.
    """
    print("Estimating voxel size...")
    if len(points) > num_samples:
        idx = np.random.choice(len(points), num_samples, replace=False)
        sample = points[idx]
    else:
        sample = points

    # Brute force NN for estimation (fast enough for 1000 points)
    # dist matrix: (N, N)
    dists = np.sqrt(np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    return np.median(min_dists)

def splat_points_to_tsdf(points, normals, voxel_size, truncation_mult=4.0, chunk_size=10000):
    """
    Splat points into a TSDF volume using sparse voxel hashing concept (dictionary).
      - voxel_size: size of each voxel in scene units (0 for auto-estimation)
      - truncation_mult: voxel size multiplier to set the truncation value for signed distance function
      - chunk_size: to vectorize, we can process points in chunks, adjust based on memory
    """
    if voxel_size <= 0:
        voxel_size = estimate_voxel_size(points)
        print(f"Estimated voxel size: {voxel_size:.4f}")
    truncation = truncation_mult * voxel_size
    voxel_r = int(np.ceil(truncation / voxel_size))
    
    # Grid limits
    min_bound = np.min(points, axis=0) - truncation
    
    # Sparse TSDF storage: key=(ix, iy, iz), value=[w_sum, d_w_sum]
    # We use a dictionary for sparse storage
    tsdf_vol = {}
    
    print(f"Splatting {len(points)} points into TSDF...")
    
    # Discretize point positions
    pt_voxels = np.floor((points - min_bound) / voxel_size).astype(int)
    
    # Define neighborhood offsets
    r_range = range(-voxel_r, voxel_r + 1)
    offsets = np.array(np.meshgrid(r_range, r_range, r_range, indexing='ij')).reshape(3, -1).T
    
    
    for i in tqdm(range(0, len(points), chunk_size), desc="Integrating"):
        end = min(i + chunk_size, len(points))
        pts_chunk = points[i:end]
        nrms_chunk = normals[i:end]
        vox_chunk = pt_voxels[i:end]
        
        # This part is tricky to fully vectorize without exploding memory if we just broadcast
        # Strategy: Iterate over offsets (constant number, e.g. 5x5x5=125) 
        # and apply to all points in chunk.
        
        for off in offsets:
            # Candidate voxel indices for the whole chunk
            cand_vox_indices = vox_chunk + off
            
            # Candidate voxel positions in world space
            cand_vox_pos = min_bound + (cand_vox_indices + 0.5) * voxel_size
            
            # Vector from point to voxel center
            # diff = voxel - point
            diff = cand_vox_pos - pts_chunk
            
            # SDF = (v - p) . n
            # Note: The prompt formulation says SDF = (v - p) . n
            # If v is outside (in front of surface), and normal points out, (v-p).n > 0.
            sdf_vals = np.sum(diff * nrms_chunk, axis=1)
            
            # Check truncation
            valid_mask = np.abs(sdf_vals) < truncation
            
            if not np.any(valid_mask):
                continue
                
            valid_indices = cand_vox_indices[valid_mask]
            valid_sdfs = sdf_vals[valid_mask]
            
            # Update TSDF
            # We can't easily vector-update a simple dict. 
            # But we can create a local hash map/list and merge?
            # Or just loop for the valid ones (should be smaller subset)
            
            # Optimization: Weighting
            # weight = 1.0 (Simple)
            weights = np.ones_like(valid_sdfs)
            
            # We need to aggregate. 
            # Since we are in Python, dict access is slow in a tight loop.
            # Faster approach: Store all updates in a list/arrays and aggregate later.
            # But memory might be an issue.
            
            # Let's try to aggregate locally in chunk then update global dict?
            # Or use a flat array of 'hashed' indices if domain is known?
            # Since we don't know the full domain size perfectly without allocating dense grid,
            # let's stick to dictionary but maybe use a flat key?
            
            # Flatten keys for dictionary
            keys = tuple(map(tuple, valid_indices))
            
            for k, sdf, w in zip(keys, valid_sdfs, weights):
                if k in tsdf_vol:
                    tsdf_vol[k][0] += w
                    tsdf_vol[k][1] += sdf * w
                else:
                    tsdf_vol[k] = [w, sdf * w] # [weight, weighted_sdf]

    return tsdf_vol, min_bound, voxel_size

def compute_tsdf_voxel(points, normals, voxel_size, truncation_mult=4.0, chunk_size=1000000):
    """
    Compute TSDF by creating a dense grid and querying nearest neighbors using KDTree.
    This is the "per-voxel" iteration method.
    """
    if voxel_size <= 0:
        voxel_size = estimate_voxel_size(points)
        print(f"Estimated voxel size: {voxel_size:.4f}")
    
    print("Building KDTree...")
    tree = cKDTree(points)
    
    truncation = truncation_mult * voxel_size
    
    # Define grid bounds
    min_bound = np.min(points, axis=0) - truncation
    max_bound = np.max(points, axis=0) + truncation
    
    # Create grid coordinates
    x_range = np.arange(min_bound[0], max_bound[0] + voxel_size, voxel_size)
    y_range = np.arange(min_bound[1], max_bound[1] + voxel_size, voxel_size)
    z_range = np.arange(min_bound[2], max_bound[2] + voxel_size, voxel_size)
    
    dims = (len(x_range), len(y_range), len(z_range))
    print(f"Grid dimensions: {dims} ({np.prod(dims)} voxels)")
    
    # Memory check/warning could be here. 
    # For very large clouds, this dense grid might consume too much RAM.
    
    print("Querying nearest neighbors for all voxels...")
    # Meshgrid with indexing='ij' corresponds to order x, y, z
    xv, yv, zv = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    
    # Flatten for query
    grid_points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)
    
    # Query KDTree
    sdf_values = np.zeros(len(grid_points), dtype=np.float32)
    
    for i in tqdm(range(0, len(grid_points), chunk_size), desc="Computing SDF"):
        end = min(i + chunk_size, len(grid_points))
        pts_chunk = grid_points[i:end]
        
        # Query nearest
        dists, indices = tree.query(pts_chunk, k=1, workers=-1)
        
        nearest_pts = points[indices]
        nearest_nrms = normals[indices]
        
        # Vector from point to voxel
        diff = pts_chunk - nearest_pts
        
        # SDF = (v - p) . n
        sdf_chunk = np.sum(diff * nearest_nrms, axis=1)
        
        sdf_values[i:end] = sdf_chunk
        
    # Reshape to grid
    sdf_grid = sdf_values.reshape(dims)
    
    # Truncate?
    # Usually TSDF implies truncation.
    # We can clip it, or just leave it since marching cubes finds 0-crossing.
    # But for "TSDF" consistency:
    sdf_grid = np.clip(sdf_grid, -truncation, truncation)
    
    return sdf_grid, min_bound, voxel_size

def extract_mesh_dense(sdf_grid, min_bound, voxel_size, output_file):
    print("Running Marching Cubes on dense grid...")
    try:
        verts, faces, normals, values = marching_cubes(sdf_grid, level=0.0, spacing=(voxel_size, voxel_size, voxel_size))
        
        # Transform vertices back to world space
        verts += min_bound
        
        print(f"Saving mesh to {output_file}...")
        save_ply(output_file, verts, faces, normals)
        
    except ValueError as e:
        print(f"Marching Cubes failed: {e}")
    except RuntimeError as e:
        print(f"Marching Cubes failed: {e}")

def extract_mesh(tsdf_vol, min_bound, voxel_size, output_file):
    if not tsdf_vol:
        print("Error: TSDF volume is empty.")
        return

    print("Converting sparse volume to dense grid...")
    keys = np.array(list(tsdf_vol.keys()))
    vals = np.array(list(tsdf_vol.values()))
    
    # Recover TSDF values: weighted_sdf / weight
    tsdf_values = vals[:, 1] / vals[:, 0]
    
    # Determine grid bounds
    min_idx = np.min(keys, axis=0)
    max_idx = np.max(keys, axis=0)
    dims = max_idx - min_idx + 1
    
    print(f"Grid dimensions: {dims}")
    
    # Allocate dense grid (pad with 1 to ensure boundaries for marching cubes)
    pad = 1
    grid_shape = tuple(dims + 2 * pad)
    sdf_grid = np.ones(grid_shape, dtype=np.float32) # Initialize with +1 (outside) or truncation?
    # Usually Initialize with truncation value (positive)
    
    # Fill grid
    # Shift indices to 0-based with padding
    shifted_keys = keys - min_idx + pad
    
    sdf_grid[shifted_keys[:, 0], shifted_keys[:, 1], shifted_keys[:, 2]] = tsdf_values
    
    print("Running Marching Cubes...")
    try:
        # Marching cubes
        verts, faces, normals, values = marching_cubes(sdf_grid, level=0.0, spacing=(voxel_size, voxel_size, voxel_size))
        
        # Transform vertices back to world space
        # Grid origin matches min_idx - pad
        grid_origin_idx = min_idx - pad
        grid_origin_pos = min_bound + grid_origin_idx * voxel_size
        
        verts += grid_origin_pos
        
        print(f"Saving mesh to {output_file}...")
        save_ply(output_file, verts, faces, normals)
        
    except ValueError as e:
        print(f"Marching Cubes failed: {e}")
    except RuntimeError as e:
        print(f"Marching Cubes failed: {e}")

def save_ply(path, vertices, faces, normals=None):
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if normals is not None:
        vertex_dtype.extend([('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    
    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_data['x'] = vertices[:, 0]
    vertex_data['y'] = vertices[:, 1]
    vertex_data['z'] = vertices[:, 2]
    
    if normals is not None:
        vertex_data['nx'] = normals[:, 0]
        vertex_data['ny'] = normals[:, 1]
        vertex_data['nz'] = normals[:, 2]
        
    face_data = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
    face_data['vertex_indices'] = faces
    
    el_vertex = PlyElement.describe(vertex_data, 'vertex')
    el_face = PlyElement.describe(face_data, 'face')
    
    PlyData([el_vertex, el_face], text=False).write(path)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct a mesh from a point cloud with normals using TSDF.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input PLY file")
    parser.add_argument("-o", "--output", type=str, default="mesh.ply", help="Path to output PLY file")
    parser.add_argument("-x", "--voxel_size", type=float, default=0.0, help="Voxel size (0 for auto-estimation)")
    parser.add_argument("-t", "--truncation_mult", type=float, default=3.0, help="Truncation multiplier (default: 3.0)")
    parser.add_argument("-c", "--chunk_size", type=int, default=100000, help="Chunk size for vectorized processing (default: 10000)")
    parser.add_argument("--method", type=str, choices=["splatting", "voxel"], default="splatting", help="Method: 'splatting' (sparse, faster) or 'voxel' (dense, KDTree)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    print(f"Loading {args.input}...")
    try:
        points, normals = load_ply(args.input)
    except Exception as e:
        print(f"Error loading points: {e}")
        return
    print(f"Loaded {len(points)} points.")

    if args.method == "voxel":
        # Check for scipy
        try:
            import scipy
        except ImportError:
            print("Error: 'voxel' method requires 'scipy'. Please install it: pip install scipy")
            return
        sdf_grid, min_bound, vs = compute_tsdf_voxel(points, normals, args.voxel_size, args.truncation_mult, args.chunk_size)
        extract_mesh_dense(sdf_grid, min_bound, vs, args.output)
    else:
        tsdf_vol, min_bound, vs = splat_points_to_tsdf(points, normals, args.voxel_size, args.truncation_mult, args.chunk_size)
        extract_mesh(tsdf_vol, min_bound, vs, args.output)

if __name__ == "__main__":
    main()
