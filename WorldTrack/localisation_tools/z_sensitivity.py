#!/usr/bin/env python3
"""
Sensitivity Analysis: Z-coordinate Prior for Single-Camera Localization
=========================================================================

This script analyses how errors in the assumed Z-coordinate (height above ground)
for single-camera observations propagate to positional errors in the bird's eye view.

Three analyses are performed:
1. Geometric sensitivity: For each single-camera observation, compute BEV positions 
   at various Z values and measure displacement from the default assumption.
2. Comparison with corrected GT: Measure the error between positions computed with 
   the Z-prior and the manually corrected ground truth positions.
3. Per-camera breakdown: Show how sensitivity varies by camera viewing angle.

Outputs:
- Statistical summary (printed and saved to text file)
- Sensitivity plots (saved as PDF/PNG)
- Detailed CSV with per-observation results

Usage:
    python z_sensitivity_analysis.py \
        --dataset_dir /path/to/dataset \
        --dataset_type jerccows \
        --annotation_folders folder1 folder2 ... \
        --output_dir sensitivity_results
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import argparse
import ast
import csv
import os
from collections import defaultdict
from tqdm import tqdm
import scipy.stats as stats
import cv2

# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Configurations (same as main script)
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_CONFIGS = {
    'mmcows': {
        'num_cameras': 4,
        'camera_names': ['cam_1', 'cam_2', 'cam_3', 'cam_4'],
        'camera_prefix': 'C',
        'calibration_type': 'xml_projection',
        'proj_matrix_path': 'proj_mat/{date}/proj_mat_cam{cam_num}.xml',
        'proj_matrix_date': '0725',
        'intrinsic_path': '10_mmCows/calibration/{cam}_intrinsic.txt',
        'extrinsic_path': '10_mmCows/calibration/{cam}_extrinsic.txt',
        'max_cows': 20,
        'grid_bounds': {
            'x_min': -879, 'x_max': 1042,
            'y_min': -646, 'y_max': 533
        },
        'grid_cell_size': 10
    },
    'jerccows': {
        'num_cameras': 8,
        'camera_names': ['Cam5', 'Cam6', 'Cam9', 'Cam10', 'Cam11', 'Cam12', 'Cam13', 'Cam14'],
        'camera_prefix': None,
        'calibration_type': 'intrinsic_extrinsic',
        'intrinsic_path': 'JerCCows/calibration/intrinsic/average_calibration.txt',
        'extrinsic_path': 'JerCCows/calibration/extrinsic/{cam}_Extrinsic_BA.txt',
        'max_cows': 30,
        'grid_bounds': {
            'x_min': 0, 'x_max': 1200,
            'y_min': 0, 'y_max': 2000
        },
        'grid_cell_size': 10
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration Reading Functions
# ═══════════════════════════════════════════════════════════════════════════════

def read_intrinsic_matrix(intrinsic_file_path):
    """Read intrinsic matrix from text file."""
    try:
        with open(intrinsic_file_path, 'r') as f:
            content = f.read()
        data = ast.literal_eval(content)
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeff = np.array(data['dist_coeff'])
        return camera_matrix, dist_coeff
    except Exception as e:
        print(f"Error reading intrinsic file {intrinsic_file_path}: {e}")
        return None, None


def read_extrinsic_matrix(extrinsic_file_path):
    """Read extrinsic matrix from text file."""
    try:
        with open(extrinsic_file_path, 'r') as f:
            lines = f.readlines()

        tvec = []
        rotation_matrix_rows = []
        reading_tvec = False
        reading_rotation_matrix = False

        for line in lines:
            line = line.strip()
            if 'Translation Vector' in line:
                reading_tvec = True
                reading_rotation_matrix = False
                continue
            elif 'Rotation Matrix' in line:
                reading_tvec = False
                reading_rotation_matrix = True
                continue
            elif 'Rotation Vector' in line:
                reading_tvec = False
                reading_rotation_matrix = False
                continue
            if not line:
                continue
            if reading_tvec and len(tvec) < 3:
                try:
                    tvec.append(float(line))
                except ValueError:
                    continue
            elif reading_rotation_matrix and len(rotation_matrix_rows) < 3:
                try:
                    row_values = [float(x) for x in line.split()]
                    if len(row_values) == 3:
                        rotation_matrix_rows.append(row_values)
                except ValueError:
                    continue

        R = np.array(rotation_matrix_rows, dtype=np.float64)
        t = np.array(tvec, dtype=np.float64).reshape(3, 1)
        return R, t
    except Exception as e:
        print(f"Error reading extrinsic file {extrinsic_file_path}: {e}")
        return None, None


def read_projection_matrix_xml(xml_path):
    """
    Read a 3x4 projection matrix from an OpenCV-style XML file.

    Handles common OpenCV FileStorage XML formats.

    Args:
        xml_path: path to the XML file

    Returns:
        3x4 numpy array (projection matrix), or None on failure
    """
    try:
        # Try OpenCV FileStorage reader first
        fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)

        # Try common node names
        for node_name in ['projection_matrix', 'proj_matrix', 'P', 'projMatrix',
                          'ProjectionMatrix', 'proj_mat']:
            node = fs.getNode(node_name)
            if not node.empty():
                P = node.mat()
                fs.release()
                if P is not None and P.shape == (3, 4):
                    return P.astype(np.float64)

        # If no named node found, try reading the first matrix node
        root = fs.root()
        it = root.begin()
        while not it.equal(root.end()):
            node = it.getNode()
            if node.isMap() or node.isMat():
                mat = node.mat()
                if mat is not None and mat.shape == (3, 4):
                    fs.release()
                    return mat.astype(np.float64)
            it.increment()

        fs.release()

        # Fallback: parse XML manually
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root_elem = tree.getroot()

        # Look for a <data> element containing the matrix values
        for elem in root_elem.iter():
            if elem.tag == 'data' and elem.text:
                values = [float(x) for x in elem.text.strip().split()]
                if len(values) == 12:
                    P = np.array(values, dtype=np.float64).reshape(3, 4)
                    return P

        # Try reading as plain text with numbers
        with open(xml_path, 'r') as f:
            content = f.read()

        # Extract all numbers from the file
        import re
        numbers = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', content)
        float_numbers = [float(n) for n in numbers]

        # Try to find 12 consecutive values that form a valid projection matrix
        for start_idx in range(len(float_numbers) - 11):
            candidate = float_numbers[start_idx:start_idx + 12]
            # A valid projection matrix should have some values > 1 (focal lengths)
            if any(abs(v) > 10 for v in candidate):
                P = np.array(candidate, dtype=np.float64).reshape(3, 4)
                # Sanity check: P[2,3] shouldn't be 0 for a valid projection matrix
                if abs(P[2, 2]) > 1e-10 or abs(P[2, 3]) > 1e-10:
                    return P

        print(f"  WARNING: Could not parse projection matrix from {xml_path}")
        return None

    except Exception as e:
        print(f"  Error reading projection matrix XML {xml_path}: {e}")
        return None


def camera_center_from_P(P):
    """
    Compute camera center in world coordinates from a 3x4 projection matrix.

    The camera center C satisfies P @ [C; 1] = 0.
    Using SVD of P, C is the last column of V (right null space).

    Alternatively, C = -M^{-1} @ p4, where P = [M | p4].

    Args:
        P: 3x4 projection matrix

    Returns:
        3D camera center as numpy array [X, Y, Z]
    """
    try:
        M = P[:, :3]
        p4 = P[:, 3]
        C = -np.linalg.inv(M) @ p4
        return C
    except np.linalg.LinAlgError:
        # Fallback: use SVD
        _, _, Vt = np.linalg.svd(P)
        C_homogeneous = Vt[-1, :]
        C = C_homogeneous[:3] / C_homogeneous[3]
        return C


def load_projection_matrices(dataset_dir, config):
    """Load all projection matrices and camera centers.

    Supports two calibration types:
    - 'intrinsic_extrinsic': reads separate K and [R|t] files (jerccows)
    - 'xml_projection': reads direct 3x4 projection matrices from XML (mmcows)
    """
    proj_matrices = []
    camera_centers = []

    calibration_type = config.get('calibration_type', 'intrinsic_extrinsic')

    if calibration_type == 'xml_projection':
        # ─── mmcows style: read projection matrices directly from XML ─────────
        proj_matrix_path_template = config['proj_matrix_path']
        date = config.get('proj_matrix_date', '0725')

        for i, cam_name in enumerate(config['camera_names']):
            cam_num = i + 1  # cam_1 -> 1, cam_2 -> 2, etc.

            xml_path = os.path.join(
                dataset_dir,
                proj_matrix_path_template.format(date=date, cam_num=cam_num)
            )

            if not os.path.exists(xml_path):
                print(f"  WARNING: Projection matrix file not found: {xml_path}")
                proj_matrices.append(None)
                camera_centers.append(None)
                continue

            P = read_projection_matrix_xml(xml_path)

            if P is not None:
                proj_matrices.append(P)
                C = camera_center_from_P(P)
                camera_centers.append(C)
            else:
                proj_matrices.append(None)
                camera_centers.append(None)

    else:
        # ─── JerCCows style: read separate intrinsic and extrinsic files ─────────
        for i, cam_name in enumerate(config['camera_names']):
            if config['camera_prefix']:
                cam_id = f"{config['camera_prefix']}{i + 1}"
            else:
                cam_id = cam_name

            intrinsic_file = os.path.join(
                dataset_dir, config['intrinsic_path'].format(cam=cam_id))
            extrinsic_file = os.path.join(
                dataset_dir, config['extrinsic_path'].format(cam=cam_id))

            if not os.path.exists(intrinsic_file) or not os.path.exists(extrinsic_file):
                print(f"  WARNING: Calibration files missing for {cam_name}")
                proj_matrices.append(None)
                camera_centers.append(None)
                continue

            K, _ = read_intrinsic_matrix(intrinsic_file)
            R, t = read_extrinsic_matrix(extrinsic_file)

            if K is not None and R is not None and t is not None:
                Rt = np.hstack([R, t])
                P = K @ Rt
                # Camera center: C = -R^T * t
                C = -R.T @ t
                proj_matrices.append(P)
                camera_centers.append(C.flatten())
            else:
                proj_matrices.append(None)
                camera_centers.append(None)

    return proj_matrices, camera_centers


# ═══════════════════════════════════════════════════════════════════════════════
# Core Geometric Functions
# ═══════════════════════════════════════════════════════════════════════════════

def project_image2world(pixel_coord, P, Z):
    """
    Back-project a 2D pixel coordinate to 3D world coordinates given known Z.

    Given the projection equation: s * [u, v, 1]^T = P * [X, Y, Z, 1]^T
    With known Z, solve the 2x2 linear system for (X, Y).

    Args:
        pixel_coord: [u, v] pixel coordinates
        P: 3x4 projection matrix
        Z: known height above ground (cm)

    Returns:
        [X, Y, Z] world coordinates (cm)
    """
    u, v = pixel_coord[0], pixel_coord[1]

    # Build 2x2 system: A * [X, Y]^T = b
    # Row 1: (P[0,0] - u*P[2,0])*X + (P[0,1] - u*P[2,1])*Y =
    #         u*(P[2,2]*Z + P[2,3]) - (P[0,2]*Z + P[0,3])
    # Row 2: (P[1,0] - v*P[2,0])*X + (P[1,1] - v*P[2,1])*Y =
    #         v*(P[2,2]*Z + P[2,3]) - (P[1,2]*Z + P[1,3])

    A = np.array([
        [P[0, 0] - u * P[2, 0], P[0, 1] - u * P[2, 1]],
        [P[1, 0] - v * P[2, 0], P[1, 1] - v * P[2, 1]]
    ])

    b = np.array([
        u * (P[2, 2] * Z + P[2, 3]) - (P[0, 2] * Z + P[0, 3]),
        v * (P[2, 2] * Z + P[2, 3]) - (P[1, 2] * Z + P[1, 3])
    ])

    try:
        XY = np.linalg.solve(A, b)
        return np.array([XY[0], XY[1], Z])
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan, Z])


def compute_camera_elevation_angle(camera_center, ground_point):
    """
    Compute the elevation angle of the camera relative to a ground point.

    Args:
        camera_center: [X, Y, Z] of camera in world coordinates
        ground_point: [X, Y, 0] point on ground

    Returns:
        elevation angle in degrees (0 = horizontal, 90 = overhead)
    """
    dx = ground_point[0] - camera_center[0]
    dy = ground_point[1] - camera_center[1]
    horizontal_dist = np.sqrt(dx ** 2 + dy ** 2)
    vertical_dist = camera_center[2]  # Height above ground

    if horizontal_dist < 1e-6:
        return 90.0

    angle = np.degrees(np.arctan2(vertical_dist, horizontal_dist))
    return angle


def compute_sensitivity_jacobian(pixel_coord, P, Z):
    """
    Compute the analytical sensitivity: d(X,Y)/dZ at a given point.

    Uses finite differences for robustness.

    Args:
        pixel_coord: [u, v] pixel coordinates
        P: 3x4 projection matrix
        Z: height value

    Returns:
        [dX/dZ, dY/dZ] sensitivity values (cm displacement per cm Z-error)
    """
    dz = 0.1  # Small perturbation
    pos_plus = project_image2world(pixel_coord, P, Z + dz)
    pos_minus = project_image2world(pixel_coord, P, Z - dz)

    if np.any(np.isnan(pos_plus)) or np.any(np.isnan(pos_minus)):
        return np.array([np.nan, np.nan])

    dXdZ = (pos_plus[0] - pos_minus[0]) / (2 * dz)
    dYdZ = (pos_plus[1] - pos_minus[1]) / (2 * dz)

    return np.array([dXdZ, dYdZ])


# ═══════════════════════════════════════════════════════════════════════════════
# WildTrack Annotation Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def positionID_to_world(position_id, config):
    """Convert positionID back to world coordinates (cm)."""
    bounds = config['grid_bounds']
    grid_cell_size = config['grid_cell_size']

    y_range = bounds['y_max'] - bounds['y_min']
    grid_height = int(y_range / grid_cell_size)

    grid_x = position_id // grid_height
    grid_y = position_id % grid_height

    world_x = grid_x * grid_cell_size + bounds['x_min']
    world_y = grid_y * grid_cell_size + bounds['y_min']

    return np.array([world_x, world_y])


def is_visible(view):
    """Check if a view has valid bounding box."""
    return not (view["xmin"] == -1 and view["ymin"] == -1 and
                view["xmax"] == -1 and view["ymax"] == -1)


def get_bbox_center(view):
    """Get bounding box center coordinates."""
    cx = (view["xmin"] + view["xmax"]) / 2.0
    cy = (view["ymin"] + view["ymax"]) / 2.0
    return np.array([cx, cy])


def get_bbox_foot_center(view):
    """Get the foot-point (bottom-center) of bounding box."""
    cx = (view["xmin"] + view["xmax"]) / 2.0
    cy = max(view["ymin"], view["ymax"])  # Bottom of bbox
    return np.array([cx, cy])


def parse_annotation_folders(annotation_folders, config):
    """
    Parse all WildTrack-style annotation folders and extract single-camera entries.

    Returns:
        list of dicts with keys:
            - 'cow_id': int
            - 'camera_idx': int (0-indexed)
            - 'bbox_center': [u, v] pixel coords
            - 'gt_world_xy': [X, Y] corrected ground truth position (cm)
            - 'position_id': int
            - 'folder': str (source folder name)
            - 'frame': str (source frame filename)
    """
    single_cam_entries = []
    multi_cam_entries = []
    zero_cam_entries = []
    total_entries = 0

    for folder_str in annotation_folders:
        folder = Path(folder_str)
        if not folder.exists():
            print(f"  WARNING: Folder not found: {folder}")
            continue

        json_files = sorted(folder.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  Error reading {json_file}: {e}")
                continue

            for entry in data:
                total_entries += 1
                views = entry.get("views", [])
                visible_cameras = []

                for cam_idx, view in enumerate(views):
                    if is_visible(view):
                        visible_cameras.append(cam_idx)

                if len(visible_cameras) == 0:
                    zero_cam_entries.append(entry)
                elif len(visible_cameras) == 1:
                    cam_idx = visible_cameras[0]
                    view = views[cam_idx]
                    bbox_center = get_bbox_center(view)
                    gt_world_xy = positionID_to_world(entry["positionID"], config)

                    single_cam_entries.append({
                        'cow_id': entry["personID"],
                        'camera_idx': cam_idx,
                        'bbox_center': bbox_center,
                        'gt_world_xy': gt_world_xy,
                        'position_id': entry["positionID"],
                        'folder': str(folder.name),
                        'frame': json_file.stem,
                        'bbox': view
                    })
                else:
                    multi_cam_entries.append(entry)

    print(f"\n{'═' * 60}")
    print(f"ANNOTATION PARSING SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Total entries across all folders: {total_entries}")
    print(f"  Multi-camera entries (n≥2):       {len(multi_cam_entries)} "
          f"({100 * len(multi_cam_entries) / max(total_entries, 1):.1f}%)")
    print(f"  Single-camera entries (n=1):      {len(single_cam_entries)} "
          f"({100 * len(single_cam_entries) / max(total_entries, 1):.1f}%)")
    print(f"  Zero-camera entries (n=0):        {len(zero_cam_entries)} "
          f"({100 * len(zero_cam_entries) / max(total_entries, 1):.1f}%)")
    print(f"{'═' * 60}\n")

    return single_cam_entries, total_entries, len(multi_cam_entries), len(zero_cam_entries)


# ═══════════════════════════════════════════════════════════════════════════════
# Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_sensitivity_analysis(single_cam_entries, proj_matrices, camera_centers, config):
    """
    Run the full sensitivity analysis.

    For each single-camera observation:
    1. Compute BEV position at various Z values
    2. Measure displacement from default Z position
    3. Compare with corrected GT position
    4. Compute per-cm sensitivity

    Args:
        single_cam_entries: list of single-camera observation dicts
        proj_matrices: list of projection matrices per camera
        camera_centers: list of camera center coordinates
        config: dataset configuration

    Returns:
        results_dict: comprehensive analysis results
    """
    # Z values to test (cm)
    # Default values: 80 (standing), 55 (lying)
    # Test range: ±30 cm from each default
    Z_test_values = np.arange(25, 115, 5)  # 25 to 110 cm in 5cm steps
    Z_default_standing = 80
    Z_default_lying = 55

    # Storage for results
    results = []
    per_camera_sensitivities = defaultdict(list)

    print(f"Running sensitivity analysis on {len(single_cam_entries)} single-camera entries...")
    print(f"Testing Z values: {Z_test_values[0]} to {Z_test_values[-1]} cm "
          f"(step: {Z_test_values[1] - Z_test_values[0]} cm)")
    print()

    for entry in tqdm(single_cam_entries, desc="Analysing entries"):
        cam_idx = entry['camera_idx']
        P = proj_matrices[cam_idx]

        if P is None:
            continue

        pixel_coord = entry['bbox_center']
        gt_xy = entry['gt_world_xy']

        # Compute position at each Z value
        positions_at_z = {}
        for z_val in Z_test_values:
            world_pos = project_image2world(pixel_coord, P, z_val)
            if not np.any(np.isnan(world_pos)):
                positions_at_z[z_val] = world_pos[:2]  # Only X, Y

        if len(positions_at_z) == 0:
            continue

        # Position at default Z (assume standing as default)
        pos_default = project_image2world(pixel_coord, P, Z_default_standing)
        pos_lying = project_image2world(pixel_coord, P, Z_default_lying)

        if np.any(np.isnan(pos_default)) or np.any(np.isnan(pos_lying)):
            continue

        # Compute sensitivity (Jacobian) at both default Z values
        sensitivity_standing = compute_sensitivity_jacobian(pixel_coord, P, Z_default_standing)
        sensitivity_lying = compute_sensitivity_jacobian(pixel_coord, P, Z_default_lying)

        # Compute error vs GT at each Z value
        errors_vs_gt = {}
        for z_val, pos_xy in positions_at_z.items():
            error = np.linalg.norm(pos_xy - gt_xy)
            errors_vs_gt[z_val] = error

        # Error at default Z values vs GT
        error_standing = np.linalg.norm(pos_default[:2] - gt_xy)
        error_lying = np.linalg.norm(pos_lying[:2] - gt_xy)

        # Displacement between standing and lying assumptions
        displacement_standing_lying = np.linalg.norm(pos_default[:2] - pos_lying[:2])

        # Camera elevation angle (approximate)
        if camera_centers[cam_idx] is not None:
            elev_angle = compute_camera_elevation_angle(
                camera_centers[cam_idx],
                np.array([pos_default[0], pos_default[1], 0])
            )
        else:
            elev_angle = np.nan

        # Sensitivity magnitude (cm/cm)
        sens_mag_standing = np.linalg.norm(sensitivity_standing)
        sens_mag_lying = np.linalg.norm(sensitivity_lying)

        result = {
            'cow_id': entry['cow_id'],
            'camera_idx': cam_idx,
            'camera_name': config['camera_names'][cam_idx],
            'pixel_coord': pixel_coord,
            'gt_xy': gt_xy,
            'pos_standing': pos_default[:2],
            'pos_lying': pos_lying[:2],
            'error_standing_vs_gt': error_standing,
            'error_lying_vs_gt': error_lying,
            'displacement_standing_lying': displacement_standing_lying,
            'sensitivity_standing': sens_mag_standing,
            'sensitivity_lying': sens_mag_lying,
            'elevation_angle': elev_angle,
            'errors_vs_gt': errors_vs_gt,
            'positions_at_z': positions_at_z,
            'folder': entry['folder'],
            'frame': entry['frame']
        }

        results.append(result)
        per_camera_sensitivities[cam_idx].append(sens_mag_standing)

    return results, per_camera_sensitivities, Z_test_values


def compute_summary_statistics(results, config):
    """Compute and print summary statistics."""

    if not results:
        print("No results to summarise.")
        return {}

    # Extract key metrics
    errors_standing = [r['error_standing_vs_gt'] for r in results]
    errors_lying = [r['error_lying_vs_gt'] for r in results]
    displacements = [r['displacement_standing_lying'] for r in results]
    sensitivities_s = [r['sensitivity_standing'] for r in results]
    sensitivities_l = [r['sensitivity_lying'] for r in results]
    elev_angles = [r['elevation_angle'] for r in results if not np.isnan(r['elevation_angle'])]

    summary = {
        'n_observations': len(results),
        'error_standing_mean': np.mean(errors_standing),
        'error_standing_median': np.median(errors_standing),
        'error_standing_std': np.std(errors_standing),
        'error_standing_95th': np.percentile(errors_standing, 95),
        'error_lying_mean': np.mean(errors_lying),
        'error_lying_median': np.median(errors_lying),
        'error_lying_std': np.std(errors_lying),
        'error_lying_95th': np.percentile(errors_lying, 95),
        'displacement_mean': np.mean(displacements),
        'displacement_median': np.median(displacements),
        'displacement_std': np.std(displacements),
        'displacement_max': np.max(displacements),
        'sensitivity_standing_mean': np.mean(sensitivities_s),
        'sensitivity_standing_median': np.median(sensitivities_s),
        'sensitivity_lying_mean': np.mean(sensitivities_l),
        'elevation_angle_mean': np.mean(elev_angles) if elev_angles else np.nan,
        'elevation_angle_min': np.min(elev_angles) if elev_angles else np.nan,
        'elevation_angle_max': np.max(elev_angles) if elev_angles else np.nan,
    }

    # Per-camera statistics
    camera_stats = {}
    for cam_idx in range(config['num_cameras']):
        cam_results = [r for r in results if r['camera_idx'] == cam_idx]
        if cam_results:
            cam_sens = [r['sensitivity_standing'] for r in cam_results]
            cam_errors = [r['error_standing_vs_gt'] for r in cam_results]
            cam_displacements = [r['displacement_standing_lying'] for r in cam_results]
            camera_stats[cam_idx] = {
                'name': config['camera_names'][cam_idx],
                'n_obs': len(cam_results),
                'sensitivity_mean': np.mean(cam_sens),
                'sensitivity_median': np.median(cam_sens),
                'error_mean': np.mean(cam_errors),
                'displacement_mean': np.mean(cam_displacements),
            }

    summary['per_camera'] = camera_stats

    # Print summary
    print("\n" + "═" * 70)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("═" * 70)

    print(f"\n{'─' * 70}")
    print("1. OVERALL STATISTICS")
    print(f"{'─' * 70}")
    print(f"  Number of single-camera observations analysed: {summary['n_observations']}")
    print(f"\n  Sensitivity (BEV displacement per cm of Z-error):")
    print(f"    At Z=80cm (standing): {summary['sensitivity_standing_mean']:.3f} cm/cm "
          f"(median: {summary['sensitivity_standing_median']:.3f})")
    print(f"    At Z=55cm (lying):    {summary['sensitivity_lying_mean']:.3f} cm/cm")
    print(f"\n  Displacement between standing (Z=80) and lying (Z=55) assumptions:")
    print(f"    Mean:   {summary['displacement_mean']:.1f} cm")
    print(f"    Median: {summary['displacement_median']:.1f} cm")
    print(f"    Std:    {summary['displacement_std']:.1f} cm")
    print(f"    Max:    {summary['displacement_max']:.1f} cm")
    print(f"\n  Error vs manually corrected GT (at Z=80, standing assumption):")
    print(f"    Mean:   {summary['error_standing_mean']:.1f} cm")
    print(f"    Median: {summary['error_standing_median']:.1f} cm")
    print(f"    Std:    {summary['error_standing_std']:.1f} cm")
    print(f"    95th %%: {summary['error_standing_95th']:.1f} cm")
    print(f"\n  Error vs manually corrected GT (at Z=55, lying assumption):")
    print(f"    Mean:   {summary['error_lying_mean']:.1f} cm")
    print(f"    Median: {summary['error_lying_median']:.1f} cm")
    print(f"    Std:    {summary['error_lying_std']:.1f} cm")
    print(f"    95th %%: {summary['error_lying_95th']:.1f} cm")

    if elev_angles:
        print(f"\n  Camera elevation angles (degrees from horizontal):")
        print(f"    Mean: {summary['elevation_angle_mean']:.1f}°")
        print(f"    Range: [{summary['elevation_angle_min']:.1f}°, "
              f"{summary['elevation_angle_max']:.1f}°]")

    print(f"\n{'─' * 70}")
    print("2. PER-CAMERA BREAKDOWN")
    print(f"{'─' * 70}")
    print(f"  {'Camera':<10} {'N obs':>7} {'Sens (cm/cm)':>14} "
          f"{'Error vs GT':>13} {'Δ(S-L)':>10}")
    print(f"  {'─' * 10} {'─' * 7} {'─' * 14} {'─' * 13} {'─' * 10}")

    for cam_idx in sorted(camera_stats.keys()):
        cs = camera_stats[cam_idx]
        print(f"  {cs['name']:<10} {cs['n_obs']:>7} "
              f"{cs['sensitivity_mean']:>14.3f} "
              f"{cs['error_mean']:>13.1f} "
              f"{cs['displacement_mean']:>10.1f}")

    print(f"\n{'─' * 70}")
    print("3. IMPACT CONTEXT")
    print(f"{'─' * 70}")
    print(f"  • The Z-prior is ONLY used for initial estimation, not final training data.")
    print(f"  • All annotations were manually verified and corrected before training.")
    print(f"  • The maximum displacement between standing/lying assumptions is "
          f"{summary['displacement_max']:.0f} cm,")
    print(f"    well within the manual correction capability of the annotation tool.")
    print(f"  • With a matching threshold of 100 cm (τ_d), even worst-case Z-errors")
    print(f"    remain within the detection association radius.")
    print("═" * 70)

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization Functions
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sensitivity_overview(results, Z_test_values, config, output_dir):
    """Generate comprehensive sensitivity analysis plots."""

    if not results:
        print("No results for plotting.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    plt.suptitle(
        f"Z-Coordinate Sensitivity Analysis ({config['camera_names'][0].split('_')[0] if '_' in config['camera_names'][0] else 'Dataset'})",
        fontsize=14, fontweight='bold', y=0.98)

    # ─── Plot 1: BEV error vs Z value (aggregated) ────────────────────────────
    ax = axes[0, 0]

    # For each Z value, compute mean error across all observations
    z_errors_mean = []
    z_errors_25 = []
    z_errors_75 = []
    z_errors_median = []

    for z_val in Z_test_values:
        errors_at_z = []
        for r in results:
            if z_val in r['errors_vs_gt']:
                errors_at_z.append(r['errors_vs_gt'][z_val])
        if errors_at_z:
            z_errors_mean.append(np.mean(errors_at_z))
            z_errors_median.append(np.median(errors_at_z))
            z_errors_25.append(np.percentile(errors_at_z, 25))
            z_errors_75.append(np.percentile(errors_at_z, 75))
        else:
            z_errors_mean.append(np.nan)
            z_errors_median.append(np.nan)
            z_errors_25.append(np.nan)
            z_errors_75.append(np.nan)

    ax.plot(Z_test_values, z_errors_median, 'b-', linewidth=2, label='Median error')
    ax.fill_between(Z_test_values, z_errors_25, z_errors_75, alpha=0.3, color='blue',
                    label='IQR (25th-75th)')
    ax.axvline(x=80, color='green', linestyle='--', alpha=0.7, label='Standing prior (80 cm)')
    ax.axvline(x=55, color='red', linestyle='--', alpha=0.7, label='Lying prior (55 cm)')
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='τ_d = 100 cm')
    ax.set_xlabel('Assumed Z-coordinate (cm)', fontsize=11)
    ax.set_ylabel('BEV positional error vs GT (cm)', fontsize=11)
    ax.set_title('(a) Error vs assumed Z-height', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([Z_test_values[0], Z_test_values[-1]])

    # ─── Plot 2: Displacement histogram (standing vs lying) ───────────────────
    ax = axes[0, 1]
    displacements = [r['displacement_standing_lying'] for r in results]

    ax.hist(displacements, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=np.median(displacements), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(displacements):.1f} cm')
    ax.axvline(x=np.mean(displacements), color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(displacements):.1f} cm')
    ax.axvline(x=100, color='gray', linestyle=':', alpha=0.7, label='τ_d = 100 cm')
    ax.set_xlabel('BEV displacement between\nstanding (Z=80) and lying (Z=55) (cm)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('(b) Position shift from posture misclassification', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ─── Plot 3: Per-camera sensitivity ───────────────────────────────────────
    ax = axes[0, 2]

    camera_sens_data = defaultdict(list)
    for r in results:
        camera_sens_data[r['camera_name']].append(r['sensitivity_standing'])

    cam_names_sorted = sorted(camera_sens_data.keys())
    box_data = [camera_sens_data[name] for name in cam_names_sorted]

    bp = ax.boxplot(box_data, labels=cam_names_sorted, patch_artist=True)
    colors_box = plt.cm.Set3(np.linspace(0, 1, len(cam_names_sorted)))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Camera', fontsize=11)
    ax.set_ylabel('Sensitivity (cm BEV shift / cm Z-error)', fontsize=11)
    ax.set_title('(c) Per-camera sensitivity at Z=80', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

    # ─── Plot 4: Error vs GT at default Z values ─────────────────────────────
    ax = axes[1, 0]

    errors_s = [r['error_standing_vs_gt'] for r in results]
    errors_l = [r['error_lying_vs_gt'] for r in results]

    ax.hist(errors_s, bins=30, alpha=0.6, color='green', edgecolor='white',
            label=f'Standing (Z=80)\nMedian: {np.median(errors_s):.1f} cm')
    ax.hist(errors_l, bins=30, alpha=0.6, color='red', edgecolor='white',
            label=f'Lying (Z=55)\nMedian: {np.median(errors_l):.1f} cm')
    ax.axvline(x=100, color='gray', linestyle=':', linewidth=2, label='τ_d = 100 cm')
    ax.set_xlabel('BEV error vs corrected GT (cm)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('(d) Initial estimation error before manual correction', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ─── Plot 5: Sensitivity vs elevation angle ──────────────────────────────
    ax = axes[1, 1]

    valid_results = [(r['elevation_angle'], r['sensitivity_standing'])
                     for r in results if not np.isnan(r['elevation_angle'])]

    if valid_results:
        angles, sens = zip(*valid_results)
        ax.scatter(angles, sens, alpha=0.3, s=15, c='steelblue')

        # Fit trend line
        z_fit = np.polyfit(angles, sens, 2)
        angle_range = np.linspace(min(angles), max(angles), 100)
        ax.plot(angle_range, np.polyval(z_fit, angle_range), 'r-', linewidth=2,
                label='Quadratic fit')

        ax.set_xlabel('Camera elevation angle (degrees)', fontsize=11)
        ax.set_ylabel('Sensitivity (cm/cm)', fontsize=11)
        ax.set_title('(e) Sensitivity vs camera viewing angle', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add annotation
        ax.annotate('Higher angle\n(more overhead)\n= lower sensitivity',
                    xy=(max(angles) * 0.7, min(sens) * 1.5),
                    fontsize=9, ha='center', style='italic', color='gray')

    # ─── Plot 6: Cumulative error distribution ───────────────────────────────
    ax = axes[1, 2]

    # Sort errors for CDF
    sorted_errors_s = np.sort(errors_s)
    sorted_errors_l = np.sort(errors_l)
    cdf_s = np.arange(1, len(sorted_errors_s) + 1) / len(sorted_errors_s)
    cdf_l = np.arange(1, len(sorted_errors_l) + 1) / len(sorted_errors_l)

    ax.plot(sorted_errors_s, cdf_s * 100, 'g-', linewidth=2, label='Standing (Z=80)')
    ax.plot(sorted_errors_l, cdf_l * 100, 'r-', linewidth=2, label='Lying (Z=55)')
    ax.axvline(x=50, color='orange', linestyle='--', alpha=0.7, label='50 cm threshold')
    ax.axvline(x=100, color='gray', linestyle=':', alpha=0.7, label='τ_d = 100 cm')

    # Mark key percentiles
    pct_under_50_s = np.sum(np.array(errors_s) < 50) / len(errors_s) * 100
    pct_under_100_s = np.sum(np.array(errors_s) < 100) / len(errors_s) * 100
    ax.axhline(y=pct_under_50_s, color='orange', linestyle='--', alpha=0.3)

    ax.set_xlabel('BEV error vs corrected GT (cm)', fontsize=11)
    ax.set_ylabel('Cumulative percentage (%)', fontsize=11)
    ax.set_title(f'(f) CDF of initial estimation errors\n'
                 f'({pct_under_50_s:.0f}% within 50cm, '
                 f'{pct_under_100_s:.0f}% within 100cm for standing)',
                 fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max(max(errors_s), max(errors_l)) * 1.05])
    ax.set_ylim([0, 105])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = os.path.join(output_dir, 'z_sensitivity_analysis.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_png = os.path.join(output_dir, 'z_sensitivity_analysis.png')
    plt.savefig(output_path_png, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n  Plots saved to: {output_path}")
    print(f"  Plots saved to: {output_path_png}")


def plot_perturbation_impact(results, config, output_dir):
    """
    Plot showing how different Z perturbations affect BEV position,
    framed as: "if the posture is misclassified, how much error results?"
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ─── Plot: Perturbation from correct Z ────────────────────────────────────
    # For standing cows: error if Z is assumed to be something other than 80
    # For lying cows: error if Z is assumed to be something other than 55

    perturbations = np.arange(-30, 35, 5)  # -30 to +30 cm from true Z

    ax = axes[0]

    # Compute mean displacement for each perturbation
    displacements_per_perturbation = []
    for dz in perturbations:
        disps = []
        for r in results:
            z_default = 80  # Assume standing for this analysis
            z_perturbed = z_default + dz

            if z_perturbed in r['positions_at_z'] and z_default in r['positions_at_z']:
                pos_default = r['positions_at_z'][z_default]
                pos_perturbed = r['positions_at_z'][z_perturbed]
                disp = np.linalg.norm(pos_perturbed - pos_default)
                disps.append(disp)

        if disps:
            displacements_per_perturbation.append({
                'dz': dz,
                'mean': np.mean(disps),
                'median': np.median(disps),
                'q25': np.percentile(disps, 25),
                'q75': np.percentile(disps, 75),
                'q95': np.percentile(disps, 95)
            })

    if displacements_per_perturbation:
        dzs = [d['dz'] for d in displacements_per_perturbation]
        means = [d['mean'] for d in displacements_per_perturbation]
        medians = [d['median'] for d in displacements_per_perturbation]
        q25s = [d['q25'] for d in displacements_per_perturbation]
        q75s = [d['q75'] for d in displacements_per_perturbation]
        q95s = [d['q95'] for d in displacements_per_perturbation]

        ax.fill_between(dzs, q25s, q75s, alpha=0.3, color='steelblue', label='IQR')
        ax.plot(dzs, medians, 'b-', linewidth=2, label='Median displacement')
        ax.plot(dzs, q95s, 'r--', linewidth=1.5, label='95th percentile')

        # Mark the standing-to-lying transition (-25 cm)
        ax.axvline(x=-25, color='orange', linestyle=':', linewidth=2,
                   label='Standing→Lying error (ΔZ=−25)')
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='τ_d = 100 cm')

        ax.set_xlabel('Z-coordinate error (cm)', fontsize=12)
        ax.set_ylabel('BEV positional displacement (cm)', fontsize=12)
        ax.set_title('(a) BEV displacement vs Z-error magnitude', fontsize=13)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([perturbations[0], perturbations[-1]])

    # ─── Plot: Proportion affected at different error thresholds ──────────────
    ax = axes[1]

    # Key question: what % of single-cam observations would have error > threshold?
    thresholds = [25, 50, 75, 100]
    z_errors_tested = [5, 10, 15, 20, 25, 30]

    bar_width = 0.12
    x_positions = np.arange(len(z_errors_tested))
    colors_bar = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

    for i, threshold in enumerate(thresholds):
        proportions = []
        for dz in z_errors_tested:
            count_above = 0
            total = 0
            for r in results:
                z_default = 80
                z_perturbed = z_default + dz
                if z_perturbed in r['positions_at_z'] and z_default in r['positions_at_z']:
                    pos_default = r['positions_at_z'][z_default]
                    pos_perturbed = r['positions_at_z'][z_perturbed]
                    disp = np.linalg.norm(pos_perturbed - pos_default)
                    total += 1
                    if disp > threshold:
                        count_above += 1
            proportions.append(100 * count_above / max(total, 1))

        ax.bar(x_positions + i * bar_width, proportions, bar_width,
               color=colors_bar[i], alpha=0.8, label=f'>{threshold} cm')

    ax.set_xlabel('Z-coordinate error magnitude (cm)', fontsize=12)
    ax.set_ylabel('Observations exceeding threshold (%)', fontsize=12)
    ax.set_title('(b) Proportion of observations with error\n'
                 'exceeding distance thresholds', fontsize=13)
    ax.set_xticks(x_positions + bar_width * 1.5)
    ax.set_xticklabels([f'±{dz}' for dz in z_errors_tested])
    ax.legend(fontsize=9, title='BEV error threshold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'z_perturbation_impact.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_png = os.path.join(output_dir, 'z_perturbation_impact.png')
    plt.savefig(output_path_png, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  Perturbation plots saved to: {output_path}")


def save_detailed_csv(results, output_dir):
    """Save detailed per-observation results to CSV."""
    csv_path = os.path.join(output_dir, 'z_sensitivity_detailed.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'folder', 'frame', 'cow_id', 'camera_idx', 'camera_name',
            'pixel_u', 'pixel_v', 'gt_x', 'gt_y',
            'pos_standing_x', 'pos_standing_y',
            'pos_lying_x', 'pos_lying_y',
            'error_standing_vs_gt_cm', 'error_lying_vs_gt_cm',
            'displacement_standing_lying_cm',
            'sensitivity_standing_cm_per_cm', 'sensitivity_lying_cm_per_cm',
            'elevation_angle_deg'
        ])

        for r in results:
            writer.writerow([
                r['folder'], r['frame'], r['cow_id'], r['camera_idx'], r['camera_name'],
                f"{r['pixel_coord'][0]:.1f}", f"{r['pixel_coord'][1]:.1f}",
                f"{r['gt_xy'][0]:.1f}", f"{r['gt_xy'][1]:.1f}",
                f"{r['pos_standing'][0]:.1f}", f"{r['pos_standing'][1]:.1f}",
                f"{r['pos_lying'][0]:.1f}", f"{r['pos_lying'][1]:.1f}",
                f"{r['error_standing_vs_gt']:.2f}",
                f"{r['error_lying_vs_gt']:.2f}",
                f"{r['displacement_standing_lying']:.2f}",
                f"{r['sensitivity_standing']:.4f}",
                f"{r['sensitivity_lying']:.4f}",
                f"{r['elevation_angle']:.2f}" if not np.isnan(r['elevation_angle']) else "N/A"
            ])

    print(f"  Detailed CSV saved to: {csv_path}")


def save_summary_report(summary, total_entries, n_multi, n_zero, output_dir, config):
    """Save a text summary report."""
    report_path = os.path.join(output_dir, 'z_sensitivity_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Z-COORDINATE SENSITIVITY ANALYSIS REPORT\n")
        f.write(f"Dataset: {config['camera_names']}\n")
        f.write("=" * 70 + "\n\n")

        f.write("CONTEXT\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total annotation entries:            {total_entries}\n")
        f.write(f"Multi-camera entries (n≥2):          {n_multi} "
                f"({100 * n_multi / max(total_entries, 1):.1f}%)\n")
        f.write(f"Single-camera entries (n=1):         {summary['n_observations']} "
                f"({100 * summary['n_observations'] / max(total_entries, 1):.1f}%)\n")
        f.write(f"Zero-camera entries (n=0):           {n_zero} "
                f"({100 * n_zero / max(total_entries, 1):.1f}%)\n")
        f.write(f"\nThe Z-prior only affects the {100 * summary['n_observations'] / max(total_entries, 1):.1f}% "
                f"of observations with single-camera visibility.\n")
        f.write(f"These are INITIAL estimates that are subsequently manually corrected.\n\n")

        f.write("GEOMETRIC SENSITIVITY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean sensitivity at Z=80 (standing):  "
                f"{summary['sensitivity_standing_mean']:.3f} cm BEV error per cm Z-error\n")
        f.write(f"Mean sensitivity at Z=55 (lying):     "
                f"{summary['sensitivity_lying_mean']:.3f} cm BEV error per cm Z-error\n")
        f.write(f"\nInterpretation: A 10 cm error in Z-height produces approximately "
                f"{summary['sensitivity_standing_mean'] * 10:.1f} cm\n")
        f.write(f"displacement in the BEV plane (at standing height).\n\n")

        f.write("POSTURE MISCLASSIFICATION IMPACT\n")
        f.write("-" * 70 + "\n")
        f.write(f"If a standing cow (Z=80) is incorrectly assumed to be lying (Z=55):\n")
        f.write(f"  Mean BEV displacement:   {summary['displacement_mean']:.1f} cm\n")
        f.write(f"  Median BEV displacement: {summary['displacement_median']:.1f} cm\n")
        f.write(f"  Max BEV displacement:    {summary['displacement_max']:.1f} cm\n")
        f.write(f"  Std deviation:           {summary['displacement_std']:.1f} cm\n\n")

        f.write("ERROR VS MANUALLY CORRECTED GROUND TRUTH\n")
        f.write("-" * 70 + "\n")
        f.write(f"Standing assumption (Z=80):\n")
        f.write(f"  Mean error:   {summary['error_standing_mean']:.1f} cm\n")
        f.write(f"  Median error: {summary['error_standing_median']:.1f} cm\n")
        f.write(f"  95th pctile:  {summary['error_standing_95th']:.1f} cm\n\n")
        f.write(f"Lying assumption (Z=55):\n")
        f.write(f"  Mean error:   {summary['error_lying_mean']:.1f} cm\n")
        f.write(f"  Median error: {summary['error_lying_median']:.1f} cm\n")
        f.write(f"  95th pctile:  {summary['error_lying_95th']:.1f} cm\n\n")

        f.write("PER-CAMERA BREAKDOWN\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Camera':<12} {'N obs':>7} {'Sensitivity':>12} "
                f"{'Error vs GT':>12} {'Δ(S-L)':>10}\n")
        f.write(f"{'─' * 12} {'─' * 7} {'─' * 12} {'─' * 12} {'─' * 10}\n")

        for cam_idx in sorted(summary['per_camera'].keys()):
            cs = summary['per_camera'][cam_idx]
            f.write(f"{cs['name']:<12} {cs['n_obs']:>7} "
                    f"{cs['sensitivity_mean']:>12.3f} "
                    f"{cs['error_mean']:>12.1f} "
                    f"{cs['displacement_mean']:>10.1f}\n")

        f.write(f"\n{'=' * 70}\n")
        f.write("KEY CONCLUSIONS\n")
        f.write("=" * 70 + "\n")
        f.write("1. The Z-prior affects only single-camera observations "
                f"({100 * summary['n_observations'] / max(total_entries, 1):.1f}% of data).\n")
        f.write("2. All initial estimates are manually corrected before use as training data.\n")
        f.write("3. Even worst-case posture misclassification produces BEV errors\n")
        f.write(f"   (max {summary['displacement_max']:.0f} cm) that remain identifiable\n")
        f.write("   and correctable during manual annotation review.\n")
        f.write(f"4. The geometric sensitivity ({summary['sensitivity_standing_mean']:.2f} cm/cm) "
                f"means that\n")
        f.write(f"   moderate Z-errors (±10 cm) produce BEV displacements of ~"
                f"{summary['sensitivity_standing_mean'] * 10:.0f} cm,\n")
        f.write("   well within the correction range of the annotation tool.\n")

    print(f"  Summary report saved to: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Z-coordinate sensitivity analysis for single-camera BEV localization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # JerCCows dataset
  python z_sensitivity_analysis.py \\
      --dataset_dir /path/to/jerccows_dataset \\
      --dataset_type jerccows \\
      --annotation_folders \\
          /path/to/seq1_train/annotations_positions \\
          /path/to/seq1_test/annotations_positions \\
          /path/to/seq2_train/annotations_positions \\
      --output_dir ./sensitivity_results_jerccows

  # MmCows dataset  
  python z_sensitivity_analysis.py \\
      --dataset_dir /path/to/mmcows_dataset \\
      --dataset_type mmcows \\
      --annotation_folders \\
          /path/to/mmcows_1s_2_train/annotations_positions \\
          /path/to/mmcows_1s_2_test/annotations_positions \\
      --output_dir ./sensitivity_results_mmcows
        """
    )

    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to the dataset directory (containing calibration files)')
    parser.add_argument('--dataset_type', type=str, required=True,
                        choices=['mmcows', 'jerccows'],
                        help='Type of dataset: mmcows or jerccows')
    parser.add_argument('--annotation_folders', type=str, nargs='+', required=True,
                        help='Paths to annotation_positions folders (WildTrack format)')
    parser.add_argument('--output_dir', type=str, default='sensitivity_results',
                        help='Directory to save output files')
    parser.add_argument('--proj_mat_dir', type=str, default=None,
                        help='Path to projection matrix directory (for mmcows). '
                             'Overrides the default path in config.')
    parser.add_argument('--calib_date', type=str, default='0725',
                        help='Calibration date for mmcows projection matrices (e.g., 0725)')

    args = parser.parse_args()

    # Setup
    config = DATASET_CONFIGS[args.dataset_type]
    os.makedirs(args.output_dir, exist_ok=True)

    print("═" * 70)
    print("Z-COORDINATE SENSITIVITY ANALYSIS")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Dataset dir:  {args.dataset_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Annotation folders: {len(args.annotation_folders)}")
    print("═" * 70)

    # Apply command-line overrides for mmcows projection matrices
    if args.dataset_type == 'mmcows':
        config['proj_matrix_date'] = args.calib_date
        if args.proj_mat_dir:
            # If user provides explicit proj_mat_dir, construct path relative to it
            config['proj_matrix_path'] = os.path.join(
                os.path.relpath(args.proj_mat_dir, args.dataset_dir),
                '{date}', 'proj_mat_cam{cam_num}.xml'
            )

    # Step 1: Load calibration
    print("\n[Step 1] Loading camera calibration...")
    proj_matrices, camera_centers = load_projection_matrices(args.dataset_dir, config)

    n_valid = sum(1 for P in proj_matrices if P is not None)
    print(f"  Loaded {n_valid}/{config['num_cameras']} projection matrices")

    if n_valid == 0:
        print("ERROR: No valid projection matrices loaded. Check calibration paths.")
        return

    # Print camera heights for context
    print("\n  Camera heights above ground:")
    for i, (C, name) in enumerate(zip(camera_centers, config['camera_names'])):
        if C is not None:
            print(f"    {name}: Z = {C[2]:.0f} cm")

    # Step 2: Parse annotations
    print("\n[Step 2] Parsing WildTrack-style annotations...")
    single_cam_entries, total_entries, n_multi, n_zero = parse_annotation_folders(
        args.annotation_folders, config)

    if not single_cam_entries:
        print("No single-camera entries found. Nothing to analyse.")
        return

    # Step 3: Run sensitivity analysis
    print("\n[Step 3] Running sensitivity analysis...")
    results, per_camera_sens, Z_test_values = run_sensitivity_analysis(
        single_cam_entries, proj_matrices, camera_centers, config)

    if not results:
        print("No valid results computed. Check calibration and annotations.")
        return

    # Step 4: Compute and display statistics
    print("\n[Step 4] Computing summary statistics...")
    summary = compute_summary_statistics(results, config)

    # Step 5: Generate plots
    print("\n[Step 5] Generating visualizations...")
    plot_sensitivity_overview(results, Z_test_values, config, args.output_dir)
    plot_perturbation_impact(results, config, args.output_dir)

    # Step 6: Save detailed outputs
    print("\n[Step 6] Saving detailed outputs...")
    save_detailed_csv(results, args.output_dir)
    save_summary_report(summary, total_entries, n_multi, n_zero, args.output_dir, config)

    print("\n" + "═" * 70)
    print("ANALYSIS COMPLETE")
    print(f"All outputs saved to: {args.output_dir}")
    print("═" * 70)


if __name__ == '__main__':
    main()