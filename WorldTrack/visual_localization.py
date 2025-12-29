import cv2
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import xml.etree.ElementTree as ET

from datetime import datetime
import sys
import csv
import datetime as dt
import re

import matplotlib.patches as patches
from matplotlib.lines import Line2D

from utils.pen_model import draw_pen
from utils.projection import cal_cam_coord, project_image2world
from utils.line_geometry import cal_line_equation, cal_dist_point_to_line
from utils.AdaGrad_visual_loc import visual_localization

import pytz
from tqdm import tqdm

# Set a global time zone: Central Time
CT_time_zone = pytz.timezone('America/Chicago')


def read_bbox_labels_json(json_file_path):
    """
    Read bounding box labels from a JSON file.
    Returns numpy array of bounding box data and behaviors dictionary
    """
    # Check if the file exists
    if not os.path.exists(json_file_path):
        print(f"Warning: JSON file not found: {json_file_path}")
        return np.array([]), {}

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        bboxes = []
        behaviors = {}

        for shape in data['shapes']:
            if shape['shape_type'] == 'rectangle':
                label = shape['label']
                # Extract cow ID and posture from label (e.g., "lying_1" -> 1, "standing_6" -> 6)
                posture, cow_id_str = label.split('_')
                cow_id = int(cow_id_str)

                # Map posture to behavior code (1 for standing, 7 for lying)
                behavior = 7 if posture.lower() == 'lying' else 1
                behaviors[cow_id] = behavior

                # Get the bounding box coordinates
                x1, y1 = shape['points'][0]
                x2, y2 = shape['points'][1]

                # Calculate center, width, and height
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = abs(x2 - x1)
                height = abs(y2 - y1)

                # Add confidence (set to 1.0 as it's not provided in the JSON)
                confidence = 1.0

                bboxes.append([cow_id, center_x, center_y, width, height, confidence])

        return np.array(bboxes) if bboxes else np.array([]), behaviors
    except Exception as e:
        print(f"Error reading JSON file {json_file_path}: {e}")
        return np.array([]), {}


def read_intrinsic_matrix(intrinsic_file_path):
    """
    Read intrinsic matrix from text file.
    Expected format: Python dictionary string with 'camera_matrix' and 'dist_coeff'

    Returns:
        K: 3x3 intrinsic matrix
        dist_coeff: distortion coefficients
    """
    try:
        with open(intrinsic_file_path, 'r') as f:
            content = f.read()

        # Parse the dictionary string
        import ast
        data = ast.literal_eval(content)

        # Extract camera matrix
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeff = np.array(data['dist_coeff'])

        print(f"Loaded intrinsic from {os.path.basename(intrinsic_file_path)}")
        print(f"  Camera matrix shape: {camera_matrix.shape}")

        return camera_matrix, dist_coeff

    except Exception as e:
        print(f"Error reading intrinsic file {intrinsic_file_path}: {e}")
        return None, None


def read_extrinsic_matrix(extrinsic_file_path):
    """
    Read extrinsic matrix from text file.
    Expected format:
        Rotation Vector (rvec):
        r1
        r2
        r3

        Translation Vector (tvec):
        t1
        t2
        t3

        Rotation Matrix:
        r11 r12 r13
        r21 r22 r23
        r31 r32 r33

    Returns:
        R: 3x3 rotation matrix (read directly from file)
        t: 3x1 translation vector
    """
    try:
        with open(extrinsic_file_path, 'r') as f:
            lines = f.readlines()

        tvec = []
        rotation_matrix_rows = []

        reading_tvec = False
        reading_rotation_matrix = False

        for line in lines:
            line = line.strip()

            # Check for section headers
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

            # Skip empty lines
            if not line:
                continue

            # Parse translation vector
            if reading_tvec and len(tvec) < 3:
                try:
                    value = float(line)
                    tvec.append(value)
                except ValueError:
                    continue

            # Parse rotation matrix
            elif reading_rotation_matrix and len(rotation_matrix_rows) < 3:
                try:
                    # Split the line and convert to floats
                    row_values = [float(x) for x in line.split()]
                    if len(row_values) == 3:
                        rotation_matrix_rows.append(row_values)
                    elif len(row_values) == 1:
                        # Single value per line format
                        if len(rotation_matrix_rows) == 0:
                            rotation_matrix_rows.append([row_values[0]])
                        else:
                            rotation_matrix_rows[-1].append(row_values[0])
                            if len(rotation_matrix_rows[-1]) == 3:
                                # Start new row
                                pass
                            elif len(rotation_matrix_rows[-1]) < 3:
                                # Continue current row
                                pass
                except ValueError:
                    continue

        # Validate and construct matrices
        if len(tvec) != 3:
            raise ValueError(f"Invalid tvec length: {len(tvec)}, expected 3")

        if len(rotation_matrix_rows) != 3:
            raise ValueError(f"Invalid rotation matrix rows: {len(rotation_matrix_rows)}, expected 3")

        # Convert to numpy arrays
        R = np.array(rotation_matrix_rows, dtype=np.float64)
        t = np.array(tvec, dtype=np.float64).reshape(3, 1)

        # Validate rotation matrix shape
        if R.shape != (3, 3):
            raise ValueError(f"Invalid rotation matrix shape: {R.shape}, expected (3, 3)")

        print(f"Loaded extrinsic from {os.path.basename(extrinsic_file_path)}")
        print(f"  Rotation matrix shape: {R.shape}")
        print(f"  Translation vector shape: {t.shape}")

        # Optional: Verify that R is a valid rotation matrix (det(R) ≈ 1, R^T * R ≈ I)
        det_R = np.linalg.det(R)
        if abs(det_R - 1.0) > 0.1:
            print(f"  WARNING: Rotation matrix determinant = {det_R}, expected ≈ 1.0")

        return R, t

    except Exception as e:
        print(f"Error reading extrinsic file {extrinsic_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def read_projection_matrices(dataset_dir):
    """
    Read projection matrices from intrinsic and extrinsic calibration files.
    Constructs P = K[R|t] for each camera.

    The projection matrix P maps 3D world coordinates to 2D image coordinates:
    [u, v, w]^T = P * [X, Y, Z, 1]^T
    where (u/w, v/w) are the pixel coordinates.

    P = K[R|t] where:
    - K is the 3x3 intrinsic matrix
    - R is the 3x3 rotation matrix from world to camera coordinates
    - t is the 3x1 translation vector from world to camera coordinates

    Returns:
        proj_matrices: list of 3x4 projection matrices for each camera
    """
    proj_matrices = []
    calibration_dir = os.path.join(dataset_dir, 'calibration')

    if not os.path.exists(calibration_dir):
        print(f"ERROR: Calibration directory not found: {calibration_dir}")
        print(f"Expected path: {calibration_dir}")
        return [None, None, None, None]

    print(f"\nReading calibration files from: {calibration_dir}\n")

    for i in range(1, 5):
        intrinsic_file = os.path.join(calibration_dir, f"C{i}_intrinsic.txt")
        extrinsic_file = os.path.join(calibration_dir, f"C{i}_extrinsic.txt")

        print(f"--- Camera {i} ---")

        # Check if files exist
        if not os.path.exists(intrinsic_file):
            print(f"  ERROR: Intrinsic file not found: {intrinsic_file}")
            proj_matrices.append(None)
            continue

        if not os.path.exists(extrinsic_file):
            print(f"  ERROR: Extrinsic file not found: {extrinsic_file}")
            proj_matrices.append(None)
            continue

        # Read intrinsic matrix
        K, dist_coeff = read_intrinsic_matrix(intrinsic_file)

        # Read extrinsic matrix (rotation matrix directly from file)
        R, t = read_extrinsic_matrix(extrinsic_file)

        if K is not None and R is not None and t is not None:
            # Construct extrinsic matrix [R|t] (3x4)
            Rt = np.hstack([R, t])

            # Construct projection matrix P = K[R|t] (3x4)
            P = K @ Rt

            print(f"  Projection matrix constructed successfully")
            print(f"  P shape: {P.shape}")
            print(f"  P =")
            print(f"{P}")
            print()

            proj_matrices.append(P)
        else:
            print(f"  Failed to construct projection matrix for camera {i}")
            proj_matrices.append(None)

    return proj_matrices


def verify_calibration(dataset_dir):
    """
    Verify that calibration files are read correctly.
    """
    print("\n" + "=" * 60)
    print("CALIBRATION VERIFICATION")
    print("=" * 60)

    Proj_cam_list = read_projection_matrices(dataset_dir)

    for i, P in enumerate(Proj_cam_list):
        if P is not None:
            print(f"\nCamera {i + 1}:")
            print(f"  Projection matrix shape: {P.shape}")
            print(f"  Projection matrix:\n{P}")

            # Extract camera center from projection matrix
            cam_center = cal_cam_coord(P)
            print(f"  Camera center (world coords): {cam_center}")
        else:
            print(f"\nCamera {i + 1}: FAILED TO LOAD")

    print("\n" + "=" * 60)

def search_files(folder_path, file_format=".json"):
    """
    Search for files with a specific format in a folder.
    Returns a list of filenames.
    """
    file_names = []

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder does not exist: {folder_path}")
        return file_names

    print(f"Searching for {file_format} files in {folder_path}")

    # List files in the folder
    try:
        all_files = os.listdir(folder_path)
        print(f"Found {len(all_files)} total files in the folder")

        # Filter by file format
        file_names = [f for f in all_files if f.endswith(file_format)]
        print(f"Found {len(file_names)} {file_format} files")

        # Debug: Print first few files
        if file_names:
            print(f"First few files: {file_names[:3]}")

        return sorted(file_names)
    except Exception as e:
        print(f"Error listing files in {folder_path}: {e}")
        return []


def plot_line_3d(ax, line_eq, color, alpha):
    origin, direction = line_eq[0:3] / 100, line_eq[3:6] / 100

    ax.plot3D([origin[0], origin[0] + direction[0]],
              [origin[1], origin[1] + direction[1]],
              [origin[2], origin[2] + direction[2]], c=color, alpha=alpha)


def determine_filename_format(dataset_dir, cam_list):
    """
    Determine whether files use timestamp format or sequential format.
    Returns 'timestamp' or 'sequential'
    """
    for cam_name in cam_list:
        cam_dir = os.path.join(dataset_dir, cam_name)
        json_files = search_files(cam_dir, file_format=".json")

        if json_files:
            # Check the first file
            filename = json_files[0]

            # Sequential format (e.g., 00000001.json)
            if re.match(r'^\d{8}\.json$', filename):
                print(f"Detected sequential filename format (e.g., {filename})")
                return 'sequential'

            # Timestamp format (e.g., 1690304400_12-00-00.json)
            elif '_' in filename and re.match(r'^\d+_\d{2}-\d{2}-\d{2}\.json$', filename):
                print(f"Detected timestamp filename format (e.g., {filename})")
                return 'timestamp'

    # Default to timestamp format if can't determine
    print("Could not determine filename format, defaulting to timestamp format")
    return 'timestamp'


def extract_timestamps_from_filenames(dataset_dir, cam_list, filename_format):
    """
    Extract timestamps from filenames based on the determined format.
    Returns a sorted list of unique timestamps.
    """
    timestamp_list = set()

    for cam_name in cam_list:
        cam_dir = os.path.join(dataset_dir, cam_name)
        json_files = search_files(cam_dir, file_format=".json")

        if not json_files:
            print(f"No JSON files found in {cam_dir}")
            continue

        for json_file in json_files:
            try:
                if filename_format == 'timestamp':
                    # Extract timestamp from filename (e.g., "1690304400_12-00-00.json")
                    timestamp = int(json_file.split('_')[0])
                else:  # sequential format
                    # Extract frame number from filename (e.g., "00000001.json")
                    timestamp = int(json_file.split('.')[0])

                timestamp_list.add(timestamp)
            except Exception as e:
                print(f"Error extracting timestamp from {json_file}: {e}")

    return sorted(list(timestamp_list))


def get_json_filename(timestamp, filename_format, datetime_var=None):
    """
    Generate JSON filename based on the timestamp and format.
    """
    if filename_format == 'timestamp':
        return f'{timestamp:d}_{datetime_var.hour:02d}-{datetime_var.minute:02d}-{datetime_var.second:02d}.json'
    else:  # sequential format
        return f'{timestamp:08d}.json'


def get_image_filename(timestamp, filename_format, datetime_var=None):
    """
    Generate image filename based on the timestamp and format.
    """
    if filename_format == 'timestamp':
        return f'{timestamp:d}_{datetime_var.hour:02d}-{datetime_var.minute:02d}-{datetime_var.second:02d}.jpg'
    else:  # sequential format
        return f'{timestamp:08d}.jpg'


def get_output_filename(timestamp, filename_format, datetime_var=None):
    """
    Generate output filename based on the timestamp and format.
    """
    if filename_format == 'timestamp':
        return f'{timestamp}_{datetime_var.hour:02d}-{datetime_var.minute:02d}-{datetime_var.second:02d}.png'
    else:  # sequential format
        return f'{timestamp:08d}.png'


def process_timestamp(i, combined_timestamps, cam_list, Proj_cam_list, dataset_dir, cam_coord,
                      behaviors_dict, frame_height, frame_width, single_view, no_print, colors,
                      n_lying, n_nonlying, output_dir, filename_format, locations_file=None):
    """Process and save visualization for a single timestamp"""
    curr_timestamp = int(combined_timestamps[i])

    if filename_format == 'timestamp':
        datetime_var = datetime.fromtimestamp(curr_timestamp, CT_time_zone)
        curr_datetime = f'{datetime_var.hour:02d}-{datetime_var.minute:02d}-{datetime_var.second:02d}'
    else:
        datetime_var = None
        curr_datetime = f'frame_{curr_timestamp:08d}'

    image_name = get_image_filename(curr_timestamp, filename_format, datetime_var)
    json_filename = get_json_filename(curr_timestamp, filename_format, datetime_var)

    if not no_print:
        print(f'\n------ Processing {image_name} ------')

    ## Go through each camera
    bbox_dict_list = []
    for cam_idx, cam_name in zip(range(4), cam_list):
        cam_view_dict = {}
        proj_mat = Proj_cam_list[cam_idx]
        cam_view_dict['cam_idx'] = cam_idx
        n_rays = 0

        file_dir = os.path.join(dataset_dir, cam_name, json_filename)

        if not os.path.exists(file_dir):
            if not no_print:
                print(f"JSON file not found: {file_dir}")

        dummy_data_point = {'cow_id': -1, 'bbox': np.zeros(6), 'line_eq': np.zeros(6)}
        dummy_cam_view_dict = {'cam_idx': cam_idx, 'n_rays': 0, 'list_dict': [dummy_data_point]}

        try:
            if os.path.exists(file_dir):
                bboxes_data, curr_behaviors = read_bbox_labels_json(file_dir)

                for cow_id, behavior in curr_behaviors.items():
                    behaviors_dict[cow_id] = behavior

                if len(bboxes_data.flatten()) > 0:
                    n_rays = bboxes_data.shape[0]
                    cam_view_dict['n_rays'] = n_rays
                    list_dict = []
                    for idx, row in enumerate(bboxes_data):
                        data_point = {}
                        data_point['cow_id'] = row[0]
                        data_point['bbox_loc'] = row[1:5]
                        data_point['pixel_loc'] = row[1:3]

                        point2 = project_image2world(row[1:3], proj_mat, 30)
                        line_eq = cal_line_equation(cam_coord[cam_idx], point2)
                        data_point['line_eq'] = line_eq

                        list_dict.append(data_point)

                    cam_view_dict['list_dict'] = list_dict
                else:
                    cam_view_dict = dummy_cam_view_dict
            else:
                cam_view_dict = dummy_cam_view_dict
        except Exception as e:
            if not no_print:
                print(f"Error processing {file_dir}: {e}")
            cam_view_dict = dummy_cam_view_dict
        bbox_dict_list.append(cam_view_dict)

    ## Create the dict to store the data for each cow
    all_cows_line_set = []
    for i in range(16):
        single_cow_line_set = {'cow_id': i + 1,
                               'line_list': [],
                               'cam_idx_list': [],
                               'timestamp': curr_timestamp,
                               'location': np.empty((3)) * np.nan,
                               'pixel_list': []
                               }
        all_cows_line_set.append(single_cow_line_set)

    for cam_view_dict in bbox_dict_list:
        for single_cow_dict in all_cows_line_set:
            for single_bbox_dict in cam_view_dict['list_dict']:
                if single_cow_dict['cow_id'] == single_bbox_dict['cow_id']:
                    single_cow_dict['line_list'].append(single_bbox_dict['line_eq'])
                    single_cow_dict['cam_idx_list'].append(cam_view_dict['cam_idx'])
                    single_cow_dict['pixel_list'].append(single_bbox_dict['pixel_loc'])

    for single_cow_dict in all_cows_line_set:
        if len(single_cow_dict['line_list']) > 0:
            line_eqs = np.asarray(single_cow_dict['line_list'])

            if line_eqs.shape[0] > 1:
                nearest_point, total_distance, iter, gradient = visual_localization(line_eqs)
                nearest_point = nearest_point.astype(int)
                single_cow_dict['location'] = nearest_point
                if not no_print:
                    print(f"{single_cow_dict['cow_id']:2d}  {nearest_point}\td:{int(total_distance) / 100:.2f}\t#{iter}\tg:{gradient:.2f}")

                for i, line_eq in enumerate(line_eqs):
                    dist = int(cal_dist_point_to_line(line_eq, nearest_point))
                    if dist > 120 or i > 3:
                        cam_id = single_cow_dict['cam_idx_list'][i] + 1
                        print(f"==> Outlier: {curr_datetime} cow {single_cow_dict['cow_id']}, cam_{cam_id} ({line_eqs.shape[0]} cams)")

            elif line_eqs.shape[0] == 1 and single_view:
                cow_id = single_cow_dict['cow_id']
                behav = behaviors_dict.get(cow_id, 1)

                if behav == 7:
                    Z_set = 55
                    n_lying += 1
                else:
                    Z_set = 80
                    n_nonlying += 1

                cam_idx = int(single_cow_dict['cam_idx_list'][0])
                p_mat = Proj_cam_list[cam_idx]
                image_coord = single_cow_dict['pixel_list'][0]
                nearest_point = project_image2world(image_coord, p_mat, Z=Z_set)
                nearest_point = nearest_point.astype(int)
                single_cow_dict['location'] = nearest_point

                if not no_print:
                    print(f"{single_cow_dict['cow_id']:2d}  {nearest_point}\tbehav:{behav}")

    # Write location data to the text file
    if locations_file is not None:
        for single_cow_dict in all_cows_line_set:
            cow_id = single_cow_dict['cow_id']
            location = single_cow_dict['location']

            if not np.isnan(location[0]):
                locations_file.write(
                    f"{curr_timestamp},{cow_id},{int(location[0])},{int(location[1])},{int(location[2])}\n")

    # Collect results for this frame (for later WildTrack/GT generation)
    frame_results = {}
    for single_cow_dict in all_cows_line_set:
        cow_id = single_cow_dict['cow_id']
        location = single_cow_dict['location']
        if not np.isnan(location[0]):
            frame_results[cow_id] = location

    ## 3D Plotting ========================================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if filename_format == 'timestamp':
        ax.set_title(str(datetime_var)[0:19] + f"  {int(curr_timestamp):d}")
    else:
        ax.set_title(f"Frame {curr_timestamp:08d}")

    for single_cow_dict in all_cows_line_set:
        cow_id = single_cow_dict['cow_id']
        est_cow_loc = single_cow_dict['location'] / 100

        if cow_id < 11:
            if np.isnan(est_cow_loc[0]) == False:
                ax.scatter(est_cow_loc[0], est_cow_loc[1], est_cow_loc[2], marker='o', c=colors[cow_id], s=30)
                ax.text(est_cow_loc[0], est_cow_loc[1], est_cow_loc[2] + 0.3, f'{cow_id}', fontsize=13,
                        color=colors[cow_id], ha='center', va='bottom')

            line_eqs = single_cow_dict['line_list']
            if len(line_eqs) > 0 and cow_id != 0:
                for line in line_eqs:
                    plot_line_3d(ax, line, colors[int(cow_id)], alpha=0.35)

    draw_pen(ax, cam_coord, anchor=False, structure=False, legend=False)

    output_filename = get_output_filename(curr_timestamp, filename_format, datetime_var)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    ## 2D BEV Plotting ====================================================
    create_2d_bev_plot(frame_results, behaviors_dict, cam_coord, curr_timestamp,
                      output_dir, filename_format, datetime_var, colors)

    return n_lying, n_nonlying, frame_results


# ===============================================

def extract_bbox_from_json_file(json_file_path):
    """
    Extract bounding box information from a JSON annotation file.

    Returns:
        Dictionary mapping cow IDs to bounding box coordinates
    """
    if not os.path.exists(json_file_path):
        return {}

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        bbox_by_cow = {}
        for shape in data.get("shapes", []):
            label = shape.get("label", "")
            shape_type = shape.get("shape_type", "")

            if shape_type != "rectangle":
                continue

            parts = label.split("_")
            if len(parts) != 2:
                continue

            action, cow_str = parts
            try:
                cow_id = int(cow_str)
            except ValueError:
                continue

            pts = shape.get("points", [])
            if len(pts) != 2:
                continue

            pt1, pt2 = pts
            xmin = min(pt1[0], pt2[0])
            ymin = min(pt1[1], pt2[1])
            xmax = max(pt1[0], pt2[0])
            ymax = max(pt1[1], pt2[1])

            bbox_by_cow[cow_id] = {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            }

        return bbox_by_cow
    except Exception as e:
        print(f"Error extracting bbox from {json_file_path}: {e}")
        return {}


def create_2d_bev_plot(results, behaviors_dict, cam_coord, frame_num, output_dir,
                       filename_format, datetime_var=None, colors=None):
    """
    Create a simple 2D Bird's Eye View plot without barn elements.

    Args:
        results: Dictionary mapping cow_id to 3D location [x, y, z]
        behaviors_dict: Dictionary mapping cow_id to behavior code
        cam_coord: Array of camera coordinates
        frame_num: Frame number
        output_dir: Output directory
        filename_format: 'timestamp' or 'sequential'
        datetime_var: datetime object (for timestamp format)
        colors: List of colors for cows
    """
    if not results:
        return

    # Define behavior colors
    behavior_colors = {
        7: "red",  # lying
        1: "blue",  # standing
        0: "green"  # other/unknown
    }

    # Define camera colors
    camera_colors = ["red", "magenta", "green", "blue"]

    fig2d = plt.figure(figsize=(12, 10))
    ax2d = fig2d.add_subplot(111)

    # Plot cow positions
    for cow_id, loc in results.items():
        behavior = behaviors_dict.get(cow_id, 1)  # Default to standing
        color = behavior_colors.get(behavior, "gray")

        # Plot center point (convert from cm to meters for display if needed, or keep in cm)
        ax2d.scatter(loc[0], loc[1], marker='o', s=100, color=color, zorder=3)
        ax2d.text(loc[0] + 20, loc[1] + 20, f'{cow_id}', color='black', fontsize=10)

    # Plot camera centers
    camera_handles_2d = []
    for i, C in enumerate(cam_coord):
        color = camera_colors[i % len(camera_colors)]
        ax2d.scatter(C[0], C[1], marker='X', s=100,
                     facecolors='none', edgecolors=color, linewidth=2)
        camera_handles_2d.append(
            Line2D([0], [0], marker='X', color='w', markerfacecolor='none',
                   markeredgecolor=color, markersize=8, label=f'Cam{i + 1}')
        )

    ax2d.set_xlabel("X (cm)")
    ax2d.set_ylabel("Y (cm)")

    # Set title based on format
    if filename_format == 'timestamp':
        ax2d.set_title(f"Bird's Eye View - {str(datetime_var)[0:19]}")
    else:
        ax2d.set_title(f"Bird's Eye View - Frame {frame_num:08d}")

    ax2d.set_aspect('equal', 'box')
    ax2d.grid(True, alpha=0.3)

    # Create legends
    standing_handle = Line2D([0], [0], marker='o', color='w',
                             markerfacecolor='blue', markersize=8, label='Standing')
    lying_handle = Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='red', markersize=8, label='Lying')

    leg_behavior = ax2d.legend(handles=[standing_handle, lying_handle],
                               loc='upper right', fontsize=8, title="Behavior")
    ax2d.add_artist(leg_behavior)

    ax2d.legend(handles=camera_handles_2d, loc='upper left',
                fontsize=8, title="Cameras")

    plt.tight_layout(pad=3.0)

    # Save the figure
    if filename_format == 'timestamp':
        output_filename = f'{frame_num}_{datetime_var.hour:02d}-{datetime_var.minute:02d}-{datetime_var.second:02d}_bev.png'
    else:
        output_filename = f'{frame_num:08d}_bev.png'

    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig2d)


def generate_wildtrack_json(all_frames_data, dataset_dir, cam_list, output_dir,
                            filename_format, frame_mapping=None):
    """
    Generate JSON files in Wildtrack-like format for each frame.

    Args:
        all_frames_data: Dictionary mapping frame_num to results
        dataset_dir: Dataset directory path
        cam_list: List of camera names
        output_dir: Output directory
        filename_format: 'timestamp' or 'sequential'
        frame_mapping: Optional mapping from original to sequential frame numbers
    """
    print("\nGenerating Wildtrack-like JSON files...")

    # Create annotations_positions subfolder
    annotations_dir = os.path.join(output_dir, "annotations_positions")
    os.makedirs(annotations_dir, exist_ok=True)

    # ===== CRITICAL: Define bounds - MUST match your dataset =====
    # These are the actual world coordinate bounds of your farm in cm
    x_min_cm = -879  # Minimum X coordinate
    x_max_cm = 1042  # Maximum X coordinate
    y_min_cm = -646  # Minimum Y coordinate
    y_max_cm = 533  # Maximum Y coordinate

    grid_cell_size = 10  # cm per grid cell

    # Calculate actual grid dimensions based on bounds
    grid_width = int((x_max_cm - x_min_cm) / grid_cell_size)  # (1042 - (-879)) / 10 = 192.1 → 192
    grid_height = int((y_max_cm - y_min_cm) / grid_cell_size)  # (533 - (-646)) / 10 = 117.9 → 118

    print(f"\n=== Grid Configuration ===")
    print(f"World bounds: X=[{x_min_cm}, {x_max_cm}] cm, Y=[{y_min_cm}, {y_max_cm}] cm")
    print(f"Grid dimensions: {grid_width} × {grid_height} cells (X × Y)")
    print(f"Grid cell size: {grid_cell_size} cm")
    print("=" * 50)

    num_cameras = len(cam_list)

    # Statistics for validation
    total_cows = 0
    out_of_bounds_count = 0

    for frame_num, frame_data in all_frames_data.items():
        results = frame_data['results']

        # Get JSON files for this frame to extract bounding boxes
        if filename_format == 'timestamp':
            datetime_var = datetime.fromtimestamp(frame_num, pytz.timezone('America/Chicago'))
            json_filename = f'{frame_num:d}_{datetime_var.hour:02d}-{datetime_var.minute:02d}-{datetime_var.second:02d}.json'
        else:
            json_filename = f'{frame_num:08d}.json'

        # Extract bounding boxes from each camera
        bboxes_by_cam = {}
        for cam_name in cam_list:
            json_path = os.path.join(dataset_dir, cam_name, json_filename)
            bboxes_by_cam[cam_name] = extract_bbox_from_json_file(json_path)

        # Map frame number if mapping provided
        mapped_frame = frame_mapping.get(frame_num, frame_num) if frame_mapping else frame_num

        wildtrack_data = []

        for cow_id, cow_pos in results.items():
            total_cows += 1

            # ===== CRITICAL FIX: Account for negative coordinates =====
            # Convert world coordinates to grid coordinates
            # Subtract minimum bound before dividing by grid cell size
            grid_x = int((cow_pos[0] - x_min_cm) / grid_cell_size)
            grid_y = int((cow_pos[1] - y_min_cm) / grid_cell_size)

            # Check if out of bounds BEFORE clamping (for debugging)
            if grid_x < 0 or grid_x >= grid_width or grid_y < 0 or grid_y >= grid_height:
                out_of_bounds_count += 1
                print(f"WARNING: Cow {cow_id} in frame {frame_num} out of bounds!")
                print(f"  World pos: ({cow_pos[0]:.1f}, {cow_pos[1]:.1f}) cm")
                print(f"  Grid pos: ({grid_x}, {grid_y})")
                print(f"  Grid bounds: X=[0, {grid_width}), Y=[0, {grid_height})")

            # Clamp to valid grid range
            grid_x = max(0, min(grid_width - 1, grid_x))
            grid_y = max(0, min(grid_height - 1, grid_y))

            # Calculate positionID
            # Formula: position_id = grid_x * grid_height + grid_y
            # This MUST match get_worldgrid_from_pos() in dataset loader
            position_id = grid_x * grid_height + grid_y

            # Verification: Reverse the calculation
            verify_grid_x = position_id // grid_height
            verify_grid_y = position_id % grid_height

            if verify_grid_x != grid_x or verify_grid_y != grid_y:
                print(f"ERROR: Position ID calculation mismatch for cow {cow_id}!")
                print(f"  Original: ({grid_x}, {grid_y})")
                print(f"  Reversed: ({verify_grid_x}, {verify_grid_y})")

            # Debug output for first cow in first frame
            if total_cows == 1:
                print(f"\n=== First Cow Verification ===")
                print(f"Cow ID: {cow_id}")
                print(f"World position: ({cow_pos[0]:.1f}, {cow_pos[1]:.1f}, {cow_pos[2]:.1f}) cm")
                print(f"Grid position: ({grid_x}, {grid_y})")
                print(f"Position ID: {position_id}")

                # Verify reverse transformation
                reverse_world_x = grid_x * grid_cell_size + x_min_cm
                reverse_world_y = grid_y * grid_cell_size + y_min_cm
                print(f"Reverse world position: ({reverse_world_x:.1f}, {reverse_world_y:.1f}) cm")
                print(
                    f"Position error: ({abs(reverse_world_x - cow_pos[0]):.1f}, {abs(reverse_world_y - cow_pos[1]):.1f}) cm")
                print("=" * 50 + "\n")

            # Build views list for all cameras
            views = []
            for cam_name in cam_list:
                if cam_name in bboxes_by_cam and cow_id in bboxes_by_cam[cam_name]:
                    bbox = bboxes_by_cam[cam_name][cow_id]
                    views.append({
                        "xmin": bbox["xmin"],
                        "ymin": bbox["ymin"],
                        "xmax": bbox["xmax"],
                        "ymax": bbox["ymax"]
                    })
                else:
                    # Cow not visible in this camera
                    views.append({
                        "xmin": -1,
                        "ymin": -1,
                        "xmax": -1,
                        "ymax": -1
                    })

            # Create entry for this cow
            cow_entry = {
                "personID": int(cow_id),
                "positionID": int(position_id),
                "views": views
            }

            wildtrack_data.append(cow_entry)

        # Save JSON file for this frame
        if isinstance(mapped_frame, int):
            json_out_filename = f"{mapped_frame:08d}.json"
        else:
            json_out_filename = f"{mapped_frame}.json"

        json_path = os.path.join(annotations_dir, json_out_filename)

        with open(json_path, 'w') as f:
            json.dump(wildtrack_data, f, indent=2)

    print(f"\nGenerated {len(all_frames_data)} Wildtrack-like JSON files in {annotations_dir}")
    print(f"Total cows processed: {total_cows}")
    if out_of_bounds_count > 0:
        print(f"WARNING: {out_of_bounds_count} cows were out of bounds and clamped")


def generate_tracktacular_gt(all_frames_data, output_dir, frame_mapping=None, seq_num=1):
    """
    Generate ground truth files for TrackTacular evaluation.

    Args:
        all_frames_data: Dictionary mapping frame_num to results
        output_dir: Output directory
        frame_mapping: Optional mapping from original to sequential frame numbers
        seq_num: Sequence number for MOTA format
    """
    print("\nGenerating TrackTacular evaluation files...")

    # Create frame mapping if not provided
    if frame_mapping is None:
        frame_numbers = sorted(all_frames_data.keys())
        frame_mapping = {original: i + 1 for i, original in enumerate(frame_numbers)}

    # Create evaluations directory
    eval_dir = os.path.join(output_dir, "evaluations")
    os.makedirs(eval_dir, exist_ok=True)

    # Open files for writing
    moda_file = os.path.join(eval_dir, "gt_moda.txt")
    mota_file = os.path.join(eval_dir, "gt_mota.txt")

    with open(moda_file, 'w') as f_moda, open(mota_file, 'w') as f_mota:
        for original_frame in sorted(all_frames_data.keys()):
            mapped_frame = frame_mapping[original_frame]
            results = all_frames_data[original_frame]['results']

            for cow_id, cow_pos in results.items():
                # MODA format: [frame, x, y]
                moda_line = f"{mapped_frame},{cow_pos[0]},{cow_pos[1]}\n"
                f_moda.write(moda_line)

                # MOTA format: [seq_num, frame, id, -1, -1, -1, -1, 1, x, y, -1]
                mota_line = f"{seq_num},{mapped_frame},{cow_id},-1,-1,-1,-1,1,{cow_pos[0]},{cow_pos[1]},-1\n"
                f_mota.write(mota_line)

    print(f"Generated TrackTacular ground truth files in {eval_dir}")
    print(f"  - {moda_file}")
    print(f"  - {mota_file}")


def main(args):
    print('Dataset: ' + str(args.dataset))

    verify_calibration(args.dataset)

    global frame_height, frame_width, behaviors_dict
    global cam_list, Proj_cam_list, dataset_dir, n_lying, n_nonlying
    global cam_coord, no_print, single_view, colors

    dataset_dir = args.dataset
    frame_height = args.frame_height
    frame_width = int(frame_height * 1.6)
    no_print = args.no_print
    single_view = args.single_view
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    if single_view:
        print("Visual location with min of one view")

    cam_list = ['cam_1', 'cam_2', 'cam_3', 'cam_4']

    n_lying = 0
    n_nonlying = 0
    behaviors_dict = {}

    print("Reading projection matrices from calibration files...")
    Proj_cam_list = read_projection_matrices(dataset_dir)

    cam_coord = []
    for idx in range(4):
        proj_mat = Proj_cam_list[idx]
        cam_coord.append(cal_cam_coord(proj_mat))
        print(f"Cam {idx + 1} loc: {cal_cam_coord(proj_mat).astype(int)}")
    cam_coord = np.asarray(cam_coord).reshape((4, 3))

    filename_format = determine_filename_format(dataset_dir, cam_list)
    combined_timestamps = extract_timestamps_from_filenames(dataset_dir, cam_list, filename_format)
    print(f'Combined timestamps: {len(combined_timestamps)}')

    if combined_timestamps:
        print(f"First few timestamps: {combined_timestamps[:5]}")
    else:
        print("WARNING: No timestamps found! No images will be saved.")
        return

    colors = ['grey', 'blue', 'green', 'red', 'orange', 'black', 'purple', 'teal', 'maroon', 'hotpink', 'darkgreen',
              'aqua', 'blue', 'green', 'red', 'orange', 'black', 'purple', 'teal', 'maroon', 'hotpink']

    # Create frame mapping for sequential numbering
    frame_mapping = {ts: i + 1 for i, ts in enumerate(combined_timestamps)}

    # Dictionary to store all frame data for WildTrack/GT generation
    all_frames_data = {}

    locations_path = os.path.join(output_dir, "cow_locations.txt")
    with open(locations_path, 'w') as locations_file:
        locations_file.write("frame,id,3d_x,3d_y,3d_z\n")

        for i in tqdm(range(len(combined_timestamps)), desc="Processing frames"):
            try:
                n_lying, n_nonlying, frame_results = process_timestamp(
                    i, combined_timestamps, cam_list, Proj_cam_list, dataset_dir,
                    cam_coord, behaviors_dict, frame_height, frame_width,
                    single_view, no_print, colors, n_lying, n_nonlying, output_dir,
                    filename_format, locations_file
                )

                # Store frame results for later processing
                curr_timestamp = int(combined_timestamps[i])
                all_frames_data[curr_timestamp] = {
                    'results': frame_results,
                    'behaviors': behaviors_dict.copy()
                }

                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(combined_timestamps)} frames")

            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nFinished processing {len(combined_timestamps)} frames")
    print(f"3D plots saved to {output_dir}")
    print(f"2D BEV plots saved to {output_dir}")
    print(f"Cow locations saved to {locations_path}")
    print(f"Lying count: {n_lying}, Non-lying count: {n_nonlying}")

    # Generate WildTrack JSON files
    if all_frames_data:
        generate_wildtrack_json(all_frames_data, dataset_dir, cam_list, output_dir,
                                filename_format, frame_mapping)

        # Generate TrackTacular ground truth files
        generate_tracktacular_gt(all_frames_data, output_dir, frame_mapping, seq_num=1)
    else:
        print("\nWarning: No frame data available for WildTrack/GT generation")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CowLoc visualization with JSON annotations')
    parser.add_argument('--dataset', type=str, default='clip1_5min', help='Path to the dataset directory')
    parser.add_argument('--frame_height', type=int, default=2800, help='Height of the frame being displayed')
    parser.add_argument('--single_view', action='store_true', help='Visual location with min one view')
    parser.add_argument('--no_print', action='store_true', help='Stop printing out')
    parser.add_argument('--output_dir', type=str, default='output_frames', help='Directory to save output images')

    args = parser.parse_args()

    main(args)