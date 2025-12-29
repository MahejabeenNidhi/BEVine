import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, SpectralClustering
from collections import defaultdict
import json
import re
from pathlib import Path
import argparse
from tqdm import tqdm

# For projection
import ast

import motmetrics as mm
import math
from scipy.optimize import linear_sum_assignment


def create_frame_mapping(mot_files):
    """
    Create frame mapping from original frame numbers to sequential (1-indexed).
    This must match the mapping used in GT generation.

    Returns:
        frame_mapping: dict mapping original frame -> sequential frame
        reverse_mapping: dict mapping sequential frame -> original frame
    """
    print("\nCreating frame mapping from MOT files...")

    # Collect all unique frame numbers from all cameras
    all_frames = set()

    for mot_file in mot_files:
        if os.path.exists(mot_file):
            df = pd.read_csv(mot_file, header=None, usecols=[0])
            frames = df[0].unique()
            all_frames.update(frames)
        else:
            print(f"Warning: MOT file not found: {mot_file}")

    # Sort frames and create mapping
    sorted_frames = sorted(list(all_frames))
    frame_mapping = {int(original): i + 1 for i, original in enumerate(sorted_frames)}
    reverse_mapping = {i + 1: int(original) for i, original in enumerate(sorted_frames)}

    print(f"Frame mapping created: {len(frame_mapping)} frames")
    print(f"  Original frame range: [{min(sorted_frames)}, {max(sorted_frames)}]")
    print(f"  Mapped frame range: [1, {len(frame_mapping)}]")

    return frame_mapping, reverse_mapping

def getDistance(x1, y1, x2, y2):
    """Compute Euclidean distance between two points"""
    return math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))


def verify_frame_alignment(pred_moda_file, gt_moda_file, pred_mota_file, gt_mota_file):
    """
    Verify that prediction and GT files have aligned frame numbers and coordinates.
    """
    print("\n" + "=" * 80)
    print("FRAME ALIGNMENT VERIFICATION")
    print("=" * 80)

    # Check MODA files
    gt_moda = np.loadtxt(gt_moda_file, delimiter=',')
    pred_moda = np.loadtxt(pred_moda_file, delimiter=',')

    gt_frames = np.unique(gt_moda[:, 0].astype(int))
    pred_frames = np.unique(pred_moda[:, 0].astype(int))

    print(f"\nMODA Files:")
    print(f"  GT frames: {len(gt_frames)} (range: [{min(gt_frames)}, {max(gt_frames)}])")
    print(f"  Pred frames: {len(pred_frames)} (range: [{min(pred_frames)}, {max(pred_frames)}])")
    print(f"  Common frames: {len(np.intersect1d(gt_frames, pred_frames))}")

    # Check first few entries
    print(f"\nFirst 5 GT entries (MODA):")
    for i in range(min(5, len(gt_moda))):
        print(f"    Frame {int(gt_moda[i, 0])}: ({gt_moda[i, 1]:.1f}, {gt_moda[i, 2]:.1f})")

    print(f"\nFirst 5 Pred entries (MODA):")
    for i in range(min(5, len(pred_moda))):
        print(f"    Frame {int(pred_moda[i, 0])}: ({pred_moda[i, 1]:.1f}, {pred_moda[i, 2]:.1f})")

    # Check coordinate ranges
    print(f"\nCoordinate Ranges (MODA):")
    print(f"  GT X: [{gt_moda[:, 1].min():.1f}, {gt_moda[:, 1].max():.1f}]")
    print(f"  GT Y: [{gt_moda[:, 2].min():.1f}, {gt_moda[:, 2].max():.1f}]")
    print(f"  Pred X: [{pred_moda[:, 1].min():.1f}, {pred_moda[:, 1].max():.1f}]")
    print(f"  Pred Y: [{pred_moda[:, 2].min():.1f}, {pred_moda[:, 2].max():.1f}]")

    # Check MOTA files
    gt_mota = np.loadtxt(gt_mota_file, delimiter=',')
    pred_mota = np.loadtxt(pred_mota_file, delimiter=',')

    gt_frames_mota = np.unique(gt_mota[:, 1].astype(int))
    pred_frames_mota = np.unique(pred_mota[:, 1].astype(int))

    print(f"\nMOTA Files:")
    print(f"  GT frames: {len(gt_frames_mota)} (range: [{min(gt_frames_mota)}, {max(gt_frames_mota)}])")
    print(f"  Pred frames: {len(pred_frames_mota)} (range: [{min(pred_frames_mota)}, {max(pred_frames_mota)}])")

    # Count detections per frame
    print(f"\nDetections per frame (first 5 frames):")
    for frame in sorted(gt_frames)[:5]:
        gt_count = np.sum(gt_moda[:, 0] == frame)
        pred_count = np.sum(pred_moda[:, 0] == frame)
        print(f"  Frame {frame}: GT={gt_count}, Pred={pred_count}")

    # Check if coordinates are in same range (should both be in cm)
    x_overlap = not (gt_moda[:, 1].max() < pred_moda[:, 1].min() or
                     pred_moda[:, 1].max() < gt_moda[:, 1].min())
    y_overlap = not (gt_moda[:, 2].max() < pred_moda[:, 2].min() or
                     pred_moda[:, 2].max() < gt_moda[:, 2].min())

    print(f"\nCoordinate Overlap:")
    print(f"  X coordinates overlap: {x_overlap}")
    print(f"  Y coordinates overlap: {y_overlap}")

    if not x_overlap or not y_overlap:
        print("  WARNING: Coordinates don't overlap! Possible scale/unit mismatch!")

    print("=" * 80 + "\n")

def CLEAR_MOD_HUN(gt, det):
    """
    Compute CLEAR Detection metrics (MODA, MODP)

    Args:
        gt: Ground truth matrix [frame, x, y]
        det: Detection matrix [frame, x, y]

    Returns:
        recall, precision, MODA, MODP
    """
    td = 100 # no division because it is not going through grid cells

    if det.size == 0:
        return 0, 0, 0, 0

    # CRITICAL FIX: Use actual frame numbers, not indices
    gt_frames = gt[:, 0].astype(int)
    det_frames = det[:, 0].astype(int)

    all_frames = np.unique(np.concatenate([gt_frames, det_frames]))
    F = len(all_frames)

    # Get unique IDs if they exist (for multi-object tracking)
    # For MODA, we just need positions, so we create dummy IDs
    max_objects_per_frame = max(
        np.max(np.bincount(gt_frames)) if len(gt_frames) > 0 else 1,
        np.max(np.bincount(det_frames)) if len(det_frames) > 0 else 1
    )

    Ngt = max_objects_per_frame

    M = np.zeros((F, Ngt))
    c = np.zeros((1, F))
    fp = np.zeros((1, F))
    m = np.zeros((1, F))
    g = np.zeros((1, F))
    distances = np.inf * np.ones((F, Ngt))

    # CRITICAL FIX: Iterate over actual frame numbers
    for frame_idx, frame in enumerate(all_frames):
        # Get GT and detections for this specific frame number
        GTsInFrame = np.where(gt[:, 0] == frame)[0]
        DetsInFrame = np.where(det[:, 0] == frame)[0]

        Ngtt = len(GTsInFrame)
        Nt = len(DetsInFrame)
        g[0, frame_idx] = Ngtt

        if Ngtt > 0 and Nt > 0:
            dist = np.inf * np.ones((Ngtt, Nt))

            # Compute pairwise distances
            for o in range(Ngtt):
                GT = gt[GTsInFrame[o]][1:3]  # x, y coordinates
                for e in range(Nt):
                    E = det[DetsInFrame[e]][1:3]  # x, y coordinates
                    dist[o, e] = getDistance(GT[0], GT[1], E[0], E[1])

            # Apply threshold
            tmpai = dist.copy()
            tmpai[tmpai > td] = 1e6

            if not (tmpai == 1e6).all():
                # Hungarian algorithm for assignment
                row_ind, col_ind = linear_sum_assignment(tmpai)

                # Filter out assignments above threshold
                valid_assignments = tmpai[row_ind, col_ind] < td
                u = row_ind[valid_assignments]
                v = col_ind[valid_assignments]

                # Store matches
                for gt_idx, det_idx in zip(u, v):
                    if gt_idx < Ngt:
                        M[frame_idx, gt_idx] = det_idx + 1

                # Count matches
                curdetected = np.where(M[frame_idx, :] > 0)[0]
                c[0][frame_idx] = len(curdetected)

                # Compute distances for matched detections
                for ct in curdetected:
                    det_idx = int(M[frame_idx, ct] - 1)
                    gtX = gt[GTsInFrame[ct], 1]
                    gtY = gt[GTsInFrame[ct], 2]
                    stX = det[DetsInFrame[det_idx], 1]
                    stY = det[DetsInFrame[det_idx], 2]
                    distances[frame_idx, ct] = getDistance(gtX, gtY, stX, stY)

                fp[0][frame_idx] = Nt - c[0][frame_idx]
                m[0][frame_idx] = g[0][frame_idx] - c[0][frame_idx]
        elif Nt > 0:
            # Only detections, no GT
            fp[0][frame_idx] = Nt
            m[0][frame_idx] = 0
        elif Ngtt > 0:
            # Only GT, no detections
            fp[0][frame_idx] = 0
            m[0][frame_idx] = Ngtt

    MODP = sum(1 - distances[distances < td] / td) / np.sum(c) * 100 if np.sum(c) > 0 else 0
    MODA = (1 - ((np.sum(m) + np.sum(fp)) / np.sum(g))) * 100 if np.sum(g) > 0 else 0
    recall = np.sum(c) / np.sum(g) * 100 if np.sum(g) > 0 else 0
    precision = np.sum(c) / (np.sum(fp) + np.sum(c)) * 100 if (np.sum(fp) + np.sum(c)) > 0 else 0

    return recall, precision, MODA, MODP


def verify_coordinate_system(pred_moda_file, gt_moda_file):
    """
    Comprehensive coordinate system verification with detailed statistics.
    """
    print("\n" + "=" * 80)
    print("DETAILED COORDINATE SYSTEM VERIFICATION")
    print("=" * 80)

    gt_moda = np.loadtxt(gt_moda_file, delimiter=',')
    pred_moda = np.loadtxt(pred_moda_file, delimiter=',')

    # 1. Frame-by-frame coordinate comparison
    print("\n1. FRAME-BY-FRAME COORDINATE ANALYSIS")
    print("-" * 80)

    gt_frames = gt_moda[:, 0].astype(int)
    pred_frames = pred_moda[:, 0].astype(int)

    common_frames = np.intersect1d(gt_frames, pred_frames)

    print(f"Total common frames: {len(common_frames)}")

    # Analyze first 10 frames in detail
    for frame in common_frames[:10]:
        gt_in_frame = gt_moda[gt_moda[:, 0] == frame]
        pred_in_frame = pred_moda[pred_moda[:, 0] == frame]

        print(f"\nFrame {frame}:")
        print(f"  GT:   {len(gt_in_frame)} detections")
        print(f"  Pred: {len(pred_in_frame)} detections")

        # Compute pairwise distances
        if len(gt_in_frame) > 0 and len(pred_in_frame) > 0:
            distances = []
            for gt_det in gt_in_frame:
                for pred_det in pred_in_frame:
                    dist = np.linalg.norm(gt_det[1:3] - pred_det[1:3])
                    distances.append(dist)

            distances = np.array(distances)
            print(f"  Min distance: {distances.min():.1f} cm")
            print(f"  Mean distance: {distances.mean():.1f} cm")
            print(f"  Median distance: {np.median(distances):.1f} cm")
            print(f"  Max distance: {distances.max():.1f} cm")
            print(f"  Distances < 20cm: {np.sum(distances < 20)}")
            print(f"  Distances < 50cm: {np.sum(distances < 50)}")
            print(f"  Distances < 100cm: {np.sum(distances < 100)}")

    # 2. Overall distance distribution
    print("\n" + "-" * 80)
    print("2. OVERALL DISTANCE DISTRIBUTION")
    print("-" * 80)

    all_min_distances = []

    for frame in common_frames:
        gt_in_frame = gt_moda[gt_moda[:, 0] == frame]
        pred_in_frame = pred_moda[pred_moda[:, 0] == frame]

        if len(gt_in_frame) > 0 and len(pred_in_frame) > 0:
            # For each GT, find closest prediction
            for gt_det in gt_in_frame:
                min_dist = float('inf')
                for pred_det in pred_in_frame:
                    dist = np.linalg.norm(gt_det[1:3] - pred_det[1:3])
                    min_dist = min(min_dist, dist)
                all_min_distances.append(min_dist)

    all_min_distances = np.array(all_min_distances)

    print(f"\nClosest prediction distances (GT perspective):")
    print(f"  Total GT detections: {len(all_min_distances)}")
    print(f"  Mean min distance: {all_min_distances.mean():.1f} cm")
    print(f"  Median min distance: {np.median(all_min_distances):.1f} cm")
    print(f"  Std min distance: {all_min_distances.std():.1f} cm")
    print(f"\nMatching at different thresholds:")
    print(
        f"  < 20cm:  {np.sum(all_min_distances < 20)} ({np.sum(all_min_distances < 20) / len(all_min_distances) * 100:.1f}%)")
    print(
        f"  < 50cm:  {np.sum(all_min_distances < 50)} ({np.sum(all_min_distances < 50) / len(all_min_distances) * 100:.1f}%)")
    print(
        f"  < 100cm: {np.sum(all_min_distances < 100)} ({np.sum(all_min_distances < 100) / len(all_min_distances) * 100:.1f}%)")
    print(
        f"  < 200cm: {np.sum(all_min_distances < 200)} ({np.sum(all_min_distances < 200) / len(all_min_distances) * 100:.1f}%)")

    # 3. Distance histogram
    print("\n" + "-" * 80)
    print("3. DISTANCE HISTOGRAM")
    print("-" * 80)

    bins = [0, 20, 50, 100, 200, 500, 1000, float('inf')]
    hist, _ = np.histogram(all_min_distances, bins=bins)

    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        count = hist[i]
        pct = count / len(all_min_distances) * 100
        print(f"  {lower:4.0f} - {upper:4.0f} cm: {count:5d} ({pct:5.1f}%)")

    # 4. Coordinate bias check
    print("\n" + "-" * 80)
    print("4. COORDINATE BIAS CHECK")
    print("-" * 80)

    # For matched detections (< 100cm), check for systematic bias
    matched_pairs = []

    for frame in common_frames:
        gt_in_frame = gt_moda[gt_moda[:, 0] == frame]
        pred_in_frame = pred_moda[pred_moda[:, 0] == frame]

        if len(gt_in_frame) > 0 and len(pred_in_frame) > 0:
            # Greedy matching within 100cm
            used_pred = set()
            for gt_det in gt_in_frame:
                best_dist = float('inf')
                best_pred_idx = None

                for pred_idx, pred_det in enumerate(pred_in_frame):
                    if pred_idx in used_pred:
                        continue
                    dist = np.linalg.norm(gt_det[1:3] - pred_det[1:3])
                    if dist < best_dist and dist < 100:
                        best_dist = dist
                        best_pred_idx = pred_idx

                if best_pred_idx is not None:
                    pred_det = pred_in_frame[best_pred_idx]
                    matched_pairs.append({
                        'gt_x': gt_det[1],
                        'gt_y': gt_det[2],
                        'pred_x': pred_det[1],
                        'pred_y': pred_det[2],
                        'diff_x': pred_det[1] - gt_det[1],
                        'diff_y': pred_det[2] - gt_det[2],
                        'dist': best_dist
                    })
                    used_pred.add(best_pred_idx)

    if matched_pairs:
        matched_df = pd.DataFrame(matched_pairs)

        print(f"\nMatched pairs analysis ({len(matched_pairs)} pairs):")
        print(f"  Mean X difference: {matched_df['diff_x'].mean():.1f} cm (std: {matched_df['diff_x'].std():.1f})")
        print(f"  Mean Y difference: {matched_df['diff_y'].mean():.1f} cm (std: {matched_df['diff_y'].std():.1f})")
        print(f"  Mean distance: {matched_df['dist'].mean():.1f} cm")

        # Check for systematic bias
        if abs(matched_df['diff_x'].mean()) > 20:
            print(f"  ⚠️  WARNING: Systematic X bias detected!")
        if abs(matched_df['diff_y'].mean()) > 20:
            print(f"  ⚠️  WARNING: Systematic Y bias detected!")

    # 5. Per-cluster analysis
    print("\n" + "-" * 80)
    print("5. DETECTION COUNT ANALYSIS")
    print("-" * 80)

    detection_counts = []
    for frame in common_frames:
        gt_count = np.sum(gt_moda[:, 0] == frame)
        pred_count = np.sum(pred_moda[:, 0] == frame)
        detection_counts.append({
            'frame': frame,
            'gt_count': gt_count,
            'pred_count': pred_count,
            'diff': pred_count - gt_count
        })

    detection_df = pd.DataFrame(detection_counts)

    print(f"\nDetection count statistics:")
    print(f"  Mean GT per frame: {detection_df['gt_count'].mean():.1f}")
    print(f"  Mean Pred per frame: {detection_df['pred_count'].mean():.1f}")
    print(f"  Mean difference: {detection_df['diff'].mean():.1f}")
    print(f"  Frames with more predictions: {np.sum(detection_df['diff'] > 0)}")
    print(f"  Frames with fewer predictions: {np.sum(detection_df['diff'] < 0)}")
    print(f"  Frames with exact match: {np.sum(detection_df['diff'] == 0)}")

    print("\n" + "=" * 80 + "\n")

    return matched_df if matched_pairs else None


def visualize_matches_sample_frame(pred_moda_file, gt_moda_file, frame_num=1,
                                   output_dir='spectral_clustering_results'):
    """
    Visualize GT vs Prediction matches for a sample frame.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D

    gt_moda = np.loadtxt(gt_moda_file, delimiter=',')
    pred_moda = np.loadtxt(pred_moda_file, delimiter=',')

    gt_in_frame = gt_moda[gt_moda[:, 0] == frame_num]
    pred_in_frame = pred_moda[pred_moda[:, 0] == frame_num]

    print(f"\nVisualizing frame {frame_num}:")
    print(f"  GT: {len(gt_in_frame)} detections")
    print(f"  Pred: {len(pred_in_frame)} detections")

    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot GT positions
    for gt_det in gt_in_frame:
        ax.scatter(gt_det[1], gt_det[2], c='green', marker='o', s=200,
                   alpha=0.6, edgecolors='darkgreen', linewidths=2, label='GT' if gt_det is gt_in_frame[0] else '')
        # Draw circles at different thresholds
        circle_20 = Circle((gt_det[1], gt_det[2]), 20, fill=False,
                           edgecolor='green', linestyle='--', alpha=0.3)
        circle_50 = Circle((gt_det[1], gt_det[2]), 50, fill=False,
                           edgecolor='green', linestyle=':', alpha=0.3)
        circle_100 = Circle((gt_det[1], gt_det[2]), 100, fill=False,
                            edgecolor='green', linestyle='-.', alpha=0.2)
        ax.add_patch(circle_20)
        ax.add_patch(circle_50)
        ax.add_patch(circle_100)

    # Plot Pred positions
    for pred_det in pred_in_frame:
        ax.scatter(pred_det[1], pred_det[2], c='red', marker='x', s=200,
                   linewidths=3, label='Pred' if pred_det is pred_in_frame[0] else '')

    # Draw lines connecting closest pairs
    for gt_det in gt_in_frame:
        min_dist = float('inf')
        closest_pred = None

        for pred_det in pred_in_frame:
            dist = np.linalg.norm(gt_det[1:3] - pred_det[1:3])
            if dist < min_dist:
                min_dist = dist
                closest_pred = pred_det

        if closest_pred is not None:
            color = 'blue' if min_dist < 100 else 'orange'
            ax.plot([gt_det[1], closest_pred[1]], [gt_det[2], closest_pred[2]],
                    color=color, alpha=0.5, linewidth=1)

            # Annotate distance
            mid_x = (gt_det[1] + closest_pred[1]) / 2
            mid_y = (gt_det[2] + closest_pred[2]) / 2
            ax.text(mid_x, mid_y, f'{min_dist:.0f}cm', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    ax.set_xlabel('X (cm)', fontsize=12)
    ax.set_ylabel('Y (cm)', fontsize=12)
    ax.set_title(f'GT vs Predictions - Frame {frame_num}\n' +
                 f'Green circles: 20cm (--), 50cm (:), 100cm (-.)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(fontsize=10)

    output_path = os.path.join(output_dir, f'matching_visualization_frame_{frame_num}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def analyze_threshold_sensitivity(pred_moda_file, gt_moda_file):
    """
    Test MODA at different distance thresholds to find optimal value.
    """
    print("\n" + "=" * 80)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)

    gt_moda = np.loadtxt(gt_moda_file, delimiter=',')
    pred_moda = np.loadtxt(pred_moda_file, delimiter=',')

    thresholds = [10, 20, 30, 50, 75, 100, 150, 200]

    print(f"\nTesting MODA at different distance thresholds:\n")
    print(f"{'Threshold (cm)':<15} {'Recall (%)':<12} {'Precision (%)':<15} {'MODA (%)':<12} {'MODP (%)':<12}")
    print("-" * 80)

    for threshold in thresholds:
        # Temporarily modify the global threshold
        recall, precision, MODA, MODP = CLEAR_MOD_HUN_with_threshold(gt_moda, pred_moda, threshold)
        print(f"{threshold:<15} {recall:<12.2f} {precision:<15.2f} {MODA:<12.2f} {MODP:<12.2f}")

    print("\n" + "=" * 80 + "\n")


def CLEAR_MOD_HUN_with_threshold(gt, det, threshold_cm):
    """
    Modified CLEAR_MOD_HUN that accepts custom threshold.
    """
    td = threshold_cm  # Custom distance threshold in cm

    if det.size == 0:
        return 0, 0, 0, 0

    gt_frames = gt[:, 0].astype(int)
    det_frames = det[:, 0].astype(int)

    all_frames = np.unique(np.concatenate([gt_frames, det_frames]))
    F = len(all_frames)

    max_objects_per_frame = max(
        np.max(np.bincount(gt_frames)) if len(gt_frames) > 0 else 1,
        np.max(np.bincount(det_frames)) if len(det_frames) > 0 else 1
    )

    Ngt = max_objects_per_frame

    M = np.zeros((F, Ngt))
    c = np.zeros((1, F))
    fp = np.zeros((1, F))
    m = np.zeros((1, F))
    g = np.zeros((1, F))
    distances = np.inf * np.ones((F, Ngt))

    for frame_idx, frame in enumerate(all_frames):
        GTsInFrame = np.where(gt[:, 0] == frame)[0]
        DetsInFrame = np.where(det[:, 0] == frame)[0]

        Ngtt = len(GTsInFrame)
        Nt = len(DetsInFrame)
        g[0, frame_idx] = Ngtt

        if Ngtt > 0 and Nt > 0:
            dist = np.inf * np.ones((Ngtt, Nt))

            for o in range(Ngtt):
                GT = gt[GTsInFrame[o]][1:3]
                for e in range(Nt):
                    E = det[DetsInFrame[e]][1:3]
                    dist[o, e] = getDistance(GT[0], GT[1], E[0], E[1])

            tmpai = dist.copy()
            tmpai[tmpai > td] = 1e6

            if not (tmpai == 1e6).all():
                row_ind, col_ind = linear_sum_assignment(tmpai)

                valid_assignments = tmpai[row_ind, col_ind] < td
                u = row_ind[valid_assignments]
                v = col_ind[valid_assignments]

                for gt_idx, det_idx in zip(u, v):
                    if gt_idx < Ngt:
                        M[frame_idx, gt_idx] = det_idx + 1

                curdetected = np.where(M[frame_idx, :] > 0)[0]
                c[0][frame_idx] = len(curdetected)

                for ct in curdetected:
                    det_idx = int(M[frame_idx, ct] - 1)
                    gtX = gt[GTsInFrame[ct], 1]
                    gtY = gt[GTsInFrame[ct], 2]
                    stX = det[DetsInFrame[det_idx], 1]
                    stY = det[DetsInFrame[det_idx], 2]
                    distances[frame_idx, ct] = getDistance(gtX, gtY, stX, stY)

                fp[0][frame_idx] = Nt - c[0][frame_idx]
                m[0][frame_idx] = g[0][frame_idx] - c[0][frame_idx]
        elif Nt > 0:
            fp[0][frame_idx] = Nt
            m[0][frame_idx] = 0
        elif Ngtt > 0:
            fp[0][frame_idx] = 0
            m[0][frame_idx] = Ngtt

    MODP = sum(1 - distances[distances < td] / td) / np.sum(c) * 100 if np.sum(c) > 0 else 0
    MODA = (1 - ((np.sum(m) + np.sum(fp)) / np.sum(g))) * 100 if np.sum(g) > 0 else 0
    recall = np.sum(c) / np.sum(g) * 100 if np.sum(g) > 0 else 0
    precision = np.sum(c) / (np.sum(fp) + np.sum(c)) * 100 if (np.sum(fp) + np.sum(c)) > 0 else 0

    return recall, precision, MODA, MODP


def check_cluster_quality(tracklets, cluster_labels, output_dir='spectral_clustering_results'):
    """
    Analyze clustering quality in detail.
    """
    print("\n" + "=" * 80)
    print("DETAILED CLUSTER QUALITY ANALYSIS")
    print("=" * 80)

    # Group tracklets by cluster
    clusters = defaultdict(list)
    for i, tracklet in enumerate(tracklets):
        if tracklet.gt_id is not None:
            cluster_id = cluster_labels[i]
            clusters[cluster_id].append(tracklet)

    print(f"\nAnalyzing {len(clusters)} clusters with GT annotations:\n")

    purity_scores = []
    fragmentation_scores = []

    for cluster_id in sorted(clusters.keys()):
        cluster_tracklets = clusters[cluster_id]

        # Count GT IDs with total detections
        gt_counts = defaultdict(int)
        for tracklet in cluster_tracklets:
            gt_counts[tracklet.gt_id] += tracklet.length

        total_detections = sum(gt_counts.values())
        majority_gt = max(gt_counts, key=gt_counts.get)
        majority_count = gt_counts[majority_gt]
        purity = majority_count / total_detections

        purity_scores.append(purity)

        # Camera distribution
        cam_counts = defaultdict(int)
        for tracklet in cluster_tracklets:
            cam_counts[tracklet.cam_id + 1] += 1

        # Fragmentation: number of tracklets for majority GT ID
        majority_tracklets = sum(1 for t in cluster_tracklets if t.gt_id == majority_gt)
        fragmentation_scores.append(majority_tracklets)

        print(f"Cluster {cluster_id:2d}:")
        print(f"  Tracklets: {len(cluster_tracklets):2d}")
        print(f"  Majority GT ID: {majority_gt:2d} ({majority_count}/{total_detections} dets, purity: {purity:.1%})")
        print(f"  GT ID distribution: {dict(gt_counts)}")
        print(f"  Camera distribution: {dict(cam_counts)}")
        print(f"  Fragmentation: {majority_tracklets} tracklets for GT {majority_gt}")

        if purity < 0.7:
            print(f"  ⚠️  WARNING: Low purity cluster!")

        print()

    print("-" * 80)
    print(f"\nOverall Clustering Statistics:")
    print(f"  Mean purity: {np.mean(purity_scores):.1%}")
    print(f"  Median purity: {np.median(purity_scores):.1%}")
    print(f"  Clusters with purity > 90%: {np.sum(np.array(purity_scores) > 0.9)}/{len(purity_scores)}")
    print(f"  Clusters with purity < 70%: {np.sum(np.array(purity_scores) < 0.7)}/{len(purity_scores)}")
    print(f"  Mean fragmentation: {np.mean(fragmentation_scores):.1f} tracklets per ID")

    print("\n" + "=" * 80 + "\n")


def validate_clusters(tracklets, cluster_labels):
    """Check for impossible configurations in clusters"""
    clusters = defaultdict(list)
    for i, tracklet in enumerate(tracklets):
        clusters[cluster_labels[i]].append(tracklet)

    for cluster_id, cluster_tracklets in clusters.items():
        # Group by camera
        by_camera = defaultdict(list)
        for t in cluster_tracklets:
            by_camera[t.cam_id].append(t)

        # Check for temporal overlaps within same camera
        for cam_id, cam_tracklets in by_camera.items():
            for i, t1 in enumerate(cam_tracklets):
                for t2 in cam_tracklets[i + 1:]:
                    if t1.temporal_overlap(t2):
                        print(f"⚠️ WARNING: Cluster {cluster_id} has overlapping "
                              f"tracklets from camera {cam_id + 1}")

def mot_metrics(pred_file, gt_file, scale=0.01):
    """
    Compute MOT metrics (MOTA, MOTP, IDF1, ID switches, etc.)

    Args:
        pred_file: Path to prediction file in MOTA format
        gt_file: Path to ground truth file in MOTA format
        scale: Scale factor for distance computation (default 0.01 = 1cm)

    Returns:
        DataFrame with metrics
    """
    gt = np.loadtxt(gt_file, delimiter=',')
    dt = np.loadtxt(pred_file, delimiter=',')

    accs = []

    # Get unique sequences
    sequences = np.unique(gt[:, 0]).astype(int)

    for seq in sequences:
        acc = mm.MOTAccumulator()

        # Get frames for this sequence
        seq_gt = gt[gt[:, 0] == seq]
        seq_dt = dt[dt[:, 0] == seq]

        frames = np.unique(seq_gt[:, 1]).astype(int)

        for frame in frames:
            # Get detections for this frame
            gt_dets = seq_gt[seq_gt[:, 1] == frame][:, (2, 8, 9)]  # [id, x, y]
            dt_dets = seq_dt[seq_dt[:, 1] == frame][:, (2, 8, 9)]  # [id, x, y]

            # Compute distance matrix
            C = mm.distances.norm2squared_matrix(
                gt_dets[:, 1:3] * scale,
                dt_dets[:, 1:3] * scale,
                max_d2=1
            )
            C = np.sqrt(C)

            # Update accumulator
            acc.update(
                gt_dets[:, 0].astype('int').tolist(),
                dt_dets[:, 0].astype('int').tolist(),
                C,
                frameid=frame
            )

        accs.append(acc)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True
    )

    return summary

class Tracklet:
    """Represents a single tracklet (trajectory segment) from one camera"""

    def __init__(self, track_id, cam_id, detections):
        """
        Args:
            track_id: local track ID within the camera
            cam_id: camera index (0-3)
            detections: list of tuples (frame, bbox, conf)
        """
        self.track_id = track_id
        self.cam_id = cam_id
        self.detections = sorted(detections, key=lambda x: x[0])  # Sort by frame

        # Temporal extent
        self.start_frame = self.detections[0][0]
        self.end_frame = self.detections[-1][0]
        self.length = len(self.detections)

        # Spatial features
        self.positions_2d = []  # List of (x, y) in image coordinates
        self.positions_3d = []  # List of (X, Y, Z) in world coordinates (after projection)
        self.velocities_3d = []  # List of velocity vectors

        # Compute 2D positions (bottom-center of bbox)
        for frame, bbox, conf in self.detections:
            bb_left, bb_top, bb_width, bb_height = bbox
            center_x = bb_left + bb_width / 2
            bottom_y = bb_top + bb_height
            self.positions_2d.append((center_x, bottom_y))

        # Ground truth ID (to be filled later)
        self.gt_id = None

    def set_3d_positions(self, positions_3d):
        """Set 3D positions after projection"""
        self.positions_3d = positions_3d

        # Compute velocities
        if len(positions_3d) > 1:
            for i in range(len(positions_3d) - 1):
                vel = np.array(positions_3d[i + 1]) - np.array(positions_3d[i])
                self.velocities_3d.append(vel)

    def get_avg_position_3d(self):
        """Get average 3D position"""
        if len(self.positions_3d) == 0:
            return None
        return np.mean(self.positions_3d, axis=0)

    def get_avg_velocity_3d(self):
        """Get average 3D velocity"""
        if len(self.velocities_3d) == 0:
            return np.array([0.0, 0.0, 0.0])
        return np.mean(self.velocities_3d, axis=0)

    def get_position_at_frame(self, frame):
        """Get 3D position at a specific frame (or closest frame)"""
        if len(self.positions_3d) == 0:
            return None

        # Find closest frame
        frames = [det[0] for det in self.detections]
        idx = min(range(len(frames)), key=lambda i: abs(frames[i] - frame))
        return self.positions_3d[idx]

    def temporal_overlap(self, other):
        """Check if this tracklet overlaps in time with another"""
        return not (self.end_frame < other.start_frame or other.end_frame < self.start_frame)

    def get_common_frames(self, other):
        """Get frames where both tracklets exist"""
        self_frames = set([det[0] for det in self.detections])
        other_frames = set([det[0] for det in other.detections])
        return sorted(self_frames & other_frames)


def read_mot_file(mot_file_path, cam_id):
    """
    Read MOT format file and extract tracklets

    MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

    Returns:
        List of Tracklet objects
    """
    if not os.path.exists(mot_file_path):
        print(f"Warning: MOT file not found: {mot_file_path}")
        return []

    # Read MOT file
    df = pd.read_csv(mot_file_path, header=None,
                     names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height',
                            'conf', 'x', 'y', 'z'])

    # Group by track ID
    tracklets = []
    for track_id, group in df.groupby('id'):
        detections = []
        for _, row in group.iterrows():
            frame = int(row['frame'])
            bbox = [row['bb_left'], row['bb_top'], row['bb_width'], row['bb_height']]
            conf = row['conf']
            detections.append((frame, bbox, conf))

        tracklet = Tracklet(track_id, cam_id, detections)
        tracklets.append(tracklet)

    print(f"Camera {cam_id + 1}: Loaded {len(tracklets)} tracklets from {len(df)} detections")
    return tracklets


def read_calibration(dataset_dir, cam_id):
    """Read calibration data for projection"""
    calibration_dir = os.path.join(dataset_dir, 'calibration')

    intrinsic_file = os.path.join(calibration_dir, f"C{cam_id + 1}_intrinsic.txt")
    extrinsic_file = os.path.join(calibration_dir, f"C{cam_id + 1}_extrinsic.txt")

    # Read intrinsic
    with open(intrinsic_file, 'r') as f:
        data = ast.literal_eval(f.read())
        K = np.array(data['camera_matrix'])
        dist_coeff = np.array(data['dist_coeff'])

    # Read extrinsic
    with open(extrinsic_file, 'r') as f:
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
            tvec.append(float(line))
        elif reading_rotation_matrix and len(rotation_matrix_rows) < 3:
            row_values = [float(x) for x in line.split()]
            if len(row_values) == 3:
                rotation_matrix_rows.append(row_values)

    R = np.array(rotation_matrix_rows, dtype=np.float64)
    t = np.array(tvec, dtype=np.float64).reshape(3, 1)

    return K, R, t


def compute_ground_homography(K, R, t):
    """Compute homography for ground plane projection"""
    r1 = R[:, 0:1]
    r2 = R[:, 1:2]
    Rt_ground = np.hstack([r1, r2, t])
    H = K @ Rt_ground
    H_inv = np.linalg.inv(H)
    return H, H_inv


def project_image_to_ground(image_point, H_inv):
    """Project image point to ground plane"""
    p_img = np.array([image_point[0], image_point[1], 1.0])
    p_world_h = H_inv @ p_img
    X = p_world_h[0] / p_world_h[2]
    Y = p_world_h[1] / p_world_h[2]
    Z = 0  # Ground plane
    return np.array([X, Y, Z])


def project_tracklets_to_3d(tracklets, dataset_dir):
    """Project all tracklets to 3D world coordinates"""
    # Group tracklets by camera
    tracklets_by_cam = defaultdict(list)
    for tracklet in tracklets:
        tracklets_by_cam[tracklet.cam_id].append(tracklet)

    # Project each camera's tracklets
    for cam_id, cam_tracklets in tracklets_by_cam.items():
        print(f"Projecting camera {cam_id + 1} tracklets to 3D...")
        K, R, t = read_calibration(dataset_dir, cam_id)
        _, H_inv = compute_ground_homography(K, R, t)

        for tracklet in cam_tracklets:
            positions_3d = []
            for pos_2d in tracklet.positions_2d:
                pos_3d = project_image_to_ground(pos_2d, H_inv)
                positions_3d.append(pos_3d)
            tracklet.set_3d_positions(positions_3d)


def compute_spatial_distance(tracklet1, tracklet2):
    """
    Compute spatial distance between two tracklets
    Uses average distance at common frames if available, otherwise average positions
    """
    common_frames = tracklet1.get_common_frames(tracklet2)

    if len(common_frames) > 0:
        # Use positions at common frames
        distances = []
        for frame in common_frames:
            pos1 = tracklet1.get_position_at_frame(frame)
            pos2 = tracklet2.get_position_at_frame(frame)
            if pos1 is not None and pos2 is not None:
                dist = np.linalg.norm(pos1 - pos2)
                distances.append(dist)

        if len(distances) > 0:
            return np.mean(distances)

    # Fall back to average positions
    pos1 = tracklet1.get_avg_position_3d()
    pos2 = tracklet2.get_avg_position_3d()

    if pos1 is None or pos2 is None:
        return float('inf')

    return np.linalg.norm(pos1 - pos2)


def compute_velocity_similarity(tracklet1, tracklet2):
    """
    Compute velocity similarity (cosine similarity)
    Returns value in [0, 1], where 1 means same direction
    """
    vel1 = tracklet1.get_avg_velocity_3d()
    vel2 = tracklet2.get_avg_velocity_3d()

    # Compute magnitude
    mag1 = np.linalg.norm(vel1)
    mag2 = np.linalg.norm(vel2)

    if mag1 == 0 or mag2 == 0:
        return 0.5  # Neutral similarity if no movement

    # Cosine similarity
    cos_sim = np.dot(vel1, vel2) / (mag1 * mag2)

    # Convert to [0, 1] range
    return (cos_sim + 1) / 2


def build_similarity_matrix(tracklets, spatial_threshold=200, temporal_threshold=30):
    """
    Build similarity matrix for spectral clustering

    Returns SIMILARITY matrix (higher = more similar) for sklearn's SpectralClustering
    Invalid pairs get 0, valid pairs get exp(-combined_distance) in (0, 1]
    """
    n = len(tracklets)
    similarity_matrix = np.zeros((n, n))

    print(f"\nBuilding similarity matrix for {n} tracklets...")
    valid_pairs = 0
    invalid_pairs = 0

    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            tracklet_i = tracklets[i]
            tracklet_j = tracklets[j]

            # ===== SAME CAMERA CONSTRAINTS =====
            if tracklet_i.cam_id == tracklet_j.cam_id:  # ← ADDED CHECK
                # Cannot overlap in time
                if tracklet_i.temporal_overlap(tracklet_j):
                    similarity_matrix[i, j] = similarity_matrix[j, i] = 0
                    invalid_pairs += 1
                    continue

                # Cannot have large time gap
                time_gap = min(abs(tracklet_i.start_frame - tracklet_j.end_frame),
                               abs(tracklet_j.start_frame - tracklet_i.end_frame))

                if time_gap > temporal_threshold:
                    similarity_matrix[i, j] = similarity_matrix[j, i] = 0
                    invalid_pairs += 1
                    continue

            # ===== SPATIAL DISTANCE CONSTRAINT =====
            # (Applies to both same-camera and cross-camera)
            spatial_dist = compute_spatial_distance(tracklet_i, tracklet_j)

            if spatial_dist > spatial_threshold:
                similarity_matrix[i, j] = similarity_matrix[j, i] = 0
                invalid_pairs += 1
                continue

            # ===== COMPUTE SIMILARITY =====
            spatial_scale = 50.0
            velocity_sim = compute_velocity_similarity(tracklet_i, tracklet_j)
            velocity_distance = 1.0 - velocity_sim

            combined_distance = (0.7 * spatial_dist / spatial_scale +
                                 0.3 * velocity_distance)

            similarity = np.exp(-combined_distance)
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
            valid_pairs += 1

    print(f"Similarity matrix built:")
    print(f"  Valid pairs: {valid_pairs}")
    print(f"  Invalid pairs: {invalid_pairs}")

    return similarity_matrix


def spectral_clustering_sklearn(similarity_matrix, n_clusters=16):
    """
    Perform spectral clustering using sklearn's SpectralClustering
    (Second Script's Approach - Cleaner and More Robust)

    This replaces the manual eigenvalue decomposition approach with
    sklearn's built-in implementation.

    Args:
        similarity_matrix: Pre-computed similarity/distance matrix
        n_clusters: Number of clusters (number of cows)

    Returns:
        labels: Cluster assignment for each tracklet
    """
    print(f"\nPerforming spectral clustering using sklearn (Second Script's Approach)...")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Affinity: precomputed (using custom similarity matrix)")
    print(f"  Assign labels: kmeans (standard approach)")

    # Create SpectralClustering instance
    # Key parameters:
    # - n_clusters: number of clusters to find
    # - affinity='precomputed': use our custom similarity matrix
    # - assign_labels='kmeans': use k-means to assign final labels (standard)
    # - random_state=42: for reproducibility
    # - n_init=20: number of k-means initializations (more robust)
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42,
        n_init=20,  # Multiple k-means runs for robustness
        verbose=True  # Show progress
    )

    # Perform clustering
    print(f"\nRunning spectral clustering...")
    labels = clustering.fit_predict(similarity_matrix)

    # Print cluster distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nClustering complete!")
    print(f"Cluster distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} tracklets")

    return labels


def load_ground_truth(dataset_dir, tracklets):
    """
    Load ground truth IDs from JSON annotations
    Match tracklets to ground truth IDs based on IoU
    """
    print("\nLoading ground truth IDs from JSON annotations...")

    cam_dirs = ['cam_1', 'cam_2', 'cam_3', 'cam_4']

    # For each tracklet, find the most common ground truth ID
    for tracklet in tqdm(tracklets, desc="Matching tracklets to GT"):
        cam_dir = os.path.join(dataset_dir, cam_dirs[tracklet.cam_id])

        if not os.path.exists(cam_dir):
            print(f"Warning: Camera directory not found: {cam_dir}")
            continue

        # Count ground truth IDs across all frames in this tracklet
        gt_id_counts = defaultdict(int)

        for frame, bbox, conf in tracklet.detections:
            # Construct JSON filename (8-digit zero-padded)
            json_filename = f"{frame:08d}.json"
            json_path = os.path.join(cam_dir, json_filename)

            if not os.path.exists(json_path):
                # Try alternative naming (no padding)
                json_filename_alt = f"{frame}.json"
                json_path_alt = os.path.join(cam_dir, json_filename_alt)

                if os.path.exists(json_path_alt):
                    json_path = json_path_alt
                else:
                    continue

            # Read JSON and find matching bbox by IoU
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Extract bbox coordinates
                bb_left, bb_top, bb_width, bb_height = bbox
                pred_x1, pred_y1 = bb_left, bb_top
                pred_x2, pred_y2 = bb_left + bb_width, bb_top + bb_height

                best_iou = 0
                best_gt_id = None

                # Check all annotations in this frame
                for shape in data.get('shapes', []):
                    if shape['shape_type'] != 'rectangle':
                        continue

                    # Extract label (format: "posture_cowID" e.g., "front_5")
                    label = shape['label']

                    # Skip if not a cow annotation
                    if '_' not in label:
                        continue

                    parts = label.split('_')
                    if len(parts) != 2:
                        continue

                    posture, cow_id_str = parts

                    try:
                        gt_id = int(cow_id_str)
                    except ValueError:
                        continue

                    # Get bbox points
                    points = shape['points']
                    if len(points) != 2:
                        continue

                    gt_x1, gt_y1 = points[0]
                    gt_x2, gt_y2 = points[1]

                    # Ensure correct order (min, max)
                    gt_x1, gt_x2 = min(gt_x1, gt_x2), max(gt_x1, gt_x2)
                    gt_y1, gt_y2 = min(gt_y1, gt_y2), max(gt_y1, gt_y2)

                    # Compute IoU
                    inter_x1 = max(pred_x1, gt_x1)
                    inter_y1 = max(pred_y1, gt_y1)
                    inter_x2 = min(pred_x2, gt_x2)
                    inter_y2 = min(pred_y2, gt_y2)

                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
                        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                        union_area = pred_area + gt_area - inter_area

                        if union_area > 0:
                            iou = inter_area / union_area

                            if iou > best_iou:
                                best_iou = iou
                                best_gt_id = gt_id

                # If we found a good match (IoU > 0.5), count it
                if best_iou > 0.5 and best_gt_id is not None:
                    gt_id_counts[best_gt_id] += 1

            except Exception as e:
                print(f"\nError reading {json_path}: {e}")
                continue

        # Assign most common ground truth ID
        if len(gt_id_counts) > 0:
            tracklet.gt_id = max(gt_id_counts, key=gt_id_counts.get)
            confidence = gt_id_counts[tracklet.gt_id] / tracklet.length

            # Optionally, only assign if confidence is high enough
            if confidence < 0.3:  # Less than 30% of frames matched
                tracklet.gt_id = None

    # Report statistics
    tracklets_with_gt = sum(1 for t in tracklets if t.gt_id is not None)
    print(f"\nMatched {tracklets_with_gt}/{len(tracklets)} tracklets to ground truth IDs")

    # Show distribution of ground truth IDs
    gt_id_distribution = defaultdict(int)
    for t in tracklets:
        if t.gt_id is not None:
            gt_id_distribution[t.gt_id] += 1

    print(f"Ground truth ID distribution: {dict(sorted(gt_id_distribution.items()))}")


def compute_idf1(tracklets, cluster_labels):
    """
    Compute IDF1 score

    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    where:
    - IDTP: Number of correctly identified detections
    - IDFP: Number of false positive detections
    - IDFN: Number of false negative detections
    """
    print("\nComputing IDF1 score...")

    # Filter tracklets with ground truth
    tracklets_with_gt = [t for t in tracklets if t.gt_id is not None]

    if len(tracklets_with_gt) == 0:
        print("Warning: No tracklets matched to ground truth!")
        return 0.0

    # Count IDTP, IDFP, IDFN
    IDTP = 0
    IDFP = 0
    IDFN = 0

    # Group tracklets by cluster
    clusters = defaultdict(list)
    for i, tracklet in enumerate(tracklets_with_gt):
        cluster_id = cluster_labels[tracklets.index(tracklet)]
        clusters[cluster_id].append(tracklet)

    # For each cluster, find the majority ground truth ID
    cluster_to_gt = {}
    for cluster_id, cluster_tracklets in clusters.items():
        gt_counts = defaultdict(int)
        for tracklet in cluster_tracklets:
            gt_counts[tracklet.gt_id] += tracklet.length

        if len(gt_counts) > 0:
            cluster_to_gt[cluster_id] = max(gt_counts, key=gt_counts.get)

    # Count correct and incorrect assignments
    for i, tracklet in enumerate(tracklets_with_gt):
        cluster_id = cluster_labels[tracklets.index(tracklet)]
        predicted_id = cluster_to_gt.get(cluster_id, -1)

        if predicted_id == tracklet.gt_id:
            IDTP += tracklet.length  # Count all detections in this tracklet
        else:
            IDFP += tracklet.length  # Misidentified

    # Count false negatives (ground truth IDs not assigned)
    gt_ids_in_predictions = set(cluster_to_gt.values())
    all_gt_ids = set(t.gt_id for t in tracklets_with_gt)

    for gt_id in all_gt_ids:
        if gt_id not in gt_ids_in_predictions:
            # Count total detections for this ground truth ID
            total_detections = sum(t.length for t in tracklets_with_gt if t.gt_id == gt_id)
            IDFN += total_detections

    # Compute IDF1
    if (2 * IDTP + IDFP + IDFN) == 0:
        IDF1 = 0.0
    else:
        IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

    print(f"\nIDF1 Metrics:")
    print(f"  IDTP: {IDTP}")
    print(f"  IDFP: {IDFP}")
    print(f"  IDFN: {IDFN}")
    print(f"  IDF1: {IDF1:.4f}")

    return IDF1


def generate_global_tracks(tracklets, cluster_labels):
    """
    Generate global tracking results from clustered tracklets.

    When multiple tracklets from different cameras exist at the same frame,
    their 3D positions are averaged.

    Args:
        tracklets: List of all tracklets
        cluster_labels: Cluster assignment for each tracklet

    Returns:
        global_tracks: Dict mapping {cluster_id: {original_frame: (x, y, z, conf)}}
    """
    print("\nGenerating global tracks from clusters...")

    global_tracks = defaultdict(lambda: defaultdict(list))

    # Group tracklets by cluster
    clusters = defaultdict(list)
    for i, tracklet in enumerate(tracklets):
        cluster_id = cluster_labels[i]
        clusters[cluster_id].append(tracklet)

    # For each cluster, collect all positions at each frame
    for cluster_id, cluster_tracklets in clusters.items():
        # Collect all (frame, position, confidence) tuples
        for tracklet in cluster_tracklets:
            for det_idx, (frame, bbox, conf) in enumerate(tracklet.detections):
                # Get 3D position for this detection
                if det_idx < len(tracklet.positions_3d):
                    pos_3d = tracklet.positions_3d[det_idx]
                    # Store with ORIGINAL frame number
                    global_tracks[cluster_id][frame].append(
                        (pos_3d[0], pos_3d[1], pos_3d[2], conf)
                    )

    # Average positions when multiple tracklets exist at same frame
    averaged_tracks = {}

    for cluster_id, frame_data in global_tracks.items():
        averaged_tracks[cluster_id] = {}

        for frame, positions in frame_data.items():
            # positions is a list of (x, y, z, conf) tuples
            positions_array = np.array(positions)

            # Compute weighted average (weighted by confidence)
            weights = positions_array[:, 3]  # confidence values
            weights = weights / np.sum(weights)  # normalize

            avg_x = np.sum(positions_array[:, 0] * weights)
            avg_y = np.sum(positions_array[:, 1] * weights)
            avg_z = np.sum(positions_array[:, 2] * weights)
            avg_conf = np.mean(positions_array[:, 3])

            # Store with original frame number
            averaged_tracks[cluster_id][frame] = (avg_x, avg_y, avg_z, avg_conf)

    # Print statistics
    print(f"Generated {len(averaged_tracks)} global tracks")
    for cluster_id, frames in sorted(averaged_tracks.items())[:5]:  # Show first 5
        frame_nums = sorted(frames.keys())
        print(f"  Cluster {cluster_id}: {len(frames)} frames, range [{min(frame_nums)}, {max(frame_nums)}]")

    return averaged_tracks


def fill_track_gaps(global_tracks, max_gap=30):
    """
    Optional: Fill small gaps in tracks using linear interpolation.

    Args:
        global_tracks: Dict mapping {cluster_id: {frame: (x, y, z, conf)}}
        max_gap: Maximum gap size to fill (frames)

    Returns:
        filled_tracks: Tracks with gaps filled
    """
    filled_tracks = {}

    for cluster_id, frames_data in global_tracks.items():
        filled_tracks[cluster_id] = dict(frames_data)  # Copy existing data

        # Get sorted frames
        frames = sorted(frames_data.keys())

        if len(frames) < 2:
            continue

        # Find and fill gaps
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            gap = frame2 - frame1

            if 1 < gap <= max_gap:
                # Interpolate
                pos1 = np.array(frames_data[frame1][:3])
                pos2 = np.array(frames_data[frame2][:3])
                conf1 = frames_data[frame1][3]
                conf2 = frames_data[frame2][3]

                for j in range(1, gap):
                    interp_frame = frame1 + j
                    alpha = j / gap

                    interp_pos = pos1 * (1 - alpha) + pos2 * alpha
                    interp_conf = conf1 * (1 - alpha) + conf2 * alpha

                    filled_tracks[cluster_id][interp_frame] = (
                        interp_pos[0], interp_pos[1], interp_pos[2], interp_conf
                    )

    return filled_tracks


def save_moda_format(global_tracks, output_file, frame_mapping=None):
    """
    Save predictions in MODA format: frame, x, y

    Args:
        global_tracks: Dict mapping {cluster_id: {frame: (x, y, z, conf)}}
        output_file: Output file path
        frame_mapping: Dict mapping original frame -> sequential frame (CRITICAL!)
    """
    print(f"\nSaving MODA format to {output_file}...")

    lines = []

    for cluster_id in sorted(global_tracks.keys()):
        frames_data = global_tracks[cluster_id]

        for original_frame in sorted(frames_data.keys()):
            x, y, z, conf = frames_data[original_frame]

            # CRITICAL FIX: Apply frame mapping
            if frame_mapping is not None:
                if original_frame not in frame_mapping:
                    print(f"Warning: Frame {original_frame} not in mapping, skipping")
                    continue
                frame = frame_mapping[original_frame]
            else:
                frame = original_frame

            # MODA format: frame, x, y (coordinates in cm)
            lines.append(f"{frame},{x:.1f},{y:.1f}\n")

    # Sort by frame number
    lines.sort(key=lambda line: int(line.split(',')[0]))

    with open(output_file, 'w') as f:
        f.writelines(lines)

    print(f"Saved {len(lines)} detections in MODA format")
    print(f"  Frame range: [{int(lines[0].split(',')[0])}, {int(lines[-1].split(',')[0])}]")


def save_mota_format(global_tracks, output_file, seq_num=1, frame_mapping=None):
    """
    Save predictions in MOTA format:
    seq_num, frame, track_id, -1, -1, -1, -1, conf, x, y, -1

    Args:
        global_tracks: Dict mapping {cluster_id: {frame: (x, y, z, conf)}}
        output_file: Output file path
        seq_num: Sequence number (default 1)
        frame_mapping: Dict mapping original frame -> sequential frame (CRITICAL!)
    """
    print(f"\nSaving MOTA format to {output_file}...")

    lines = []

    for cluster_id in sorted(global_tracks.keys()):
        frames_data = global_tracks[cluster_id]

        for original_frame in sorted(frames_data.keys()):
            x, y, z, conf = frames_data[original_frame]

            # CRITICAL FIX: Apply frame mapping
            if frame_mapping is not None:
                if original_frame not in frame_mapping:
                    continue
                frame = frame_mapping[original_frame]
            else:
                frame = original_frame

            # Format: seq_num, frame, track_id, -1, -1, -1, -1, conf, x, y, -1
            lines.append(f"{seq_num},{frame},{cluster_id},-1,-1,-1,-1,{conf:.4f},{x:.1f},{y:.1f},-1\n")

    # Sort by frame, then track_id
    lines.sort(key=lambda line: (int(line.split(',')[1]), int(line.split(',')[2])))

    with open(output_file, 'w') as f:
        f.writelines(lines)

    print(f"Saved {len(lines)} tracks in MOTA format")
    if lines:
        first_frame = int(lines[0].split(',')[1])
        last_frame = int(lines[-1].split(',')[1])
        print(f"  Frame range: [{first_frame}, {last_frame}]")


def load_gt_files(gt_moda_file, gt_mota_file):
    """
    Load ground truth files

    Returns:
        gt_moda: numpy array [frame, x, y]
        gt_mota: numpy array [seq, frame, id, -1, -1, -1, -1, 1, x, y, -1]
    """
    print(f"\nLoading ground truth files...")
    print(f"  MODA GT: {gt_moda_file}")
    print(f"  MOTA GT: {gt_mota_file}")

    gt_moda = np.loadtxt(gt_moda_file, delimiter=',')
    gt_mota = np.loadtxt(gt_mota_file, delimiter=',')

    print(f"  MODA GT: {len(gt_moda)} detections")
    print(f"  MOTA GT: {len(gt_mota)} tracks")

    return gt_moda, gt_mota


def evaluate_tracking(pred_moda_file, pred_mota_file, gt_moda_file, gt_mota_file):
    """
    Comprehensive tracking evaluation

    Args:
        pred_moda_file: Prediction file in MODA format
        pred_mota_file: Prediction file in MOTA format
        gt_moda_file: Ground truth file in MODA format
        gt_mota_file: Ground truth file in MOTA format

    Returns:
        Dictionary with all metrics
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TRACKING EVALUATION")
    print("=" * 80)

    metrics = {}

    # ========== MODA/MODP Evaluation ==========
    print("\n--- Detection Metrics (MODA/MODP) ---")

    try:
        gt_moda = np.loadtxt(gt_moda_file, delimiter=',')
        pred_moda = np.loadtxt(pred_moda_file, delimiter=',')

        recall, precision, MODA, MODP = CLEAR_MOD_HUN(gt_moda, pred_moda)

        print(f"Recall:    {recall:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"MODA:      {MODA:.2f}%")
        print(f"MODP:      {MODP:.2f}%")

        metrics['recall'] = recall
        metrics['precision'] = precision
        metrics['MODA'] = MODA
        metrics['MODP'] = MODP

    except Exception as e:
        print(f"Error computing MODA/MODP: {e}")
        metrics['MODA'] = 0
        metrics['MODP'] = 0

    # ========== MOTA/MOTP/IDF1 Evaluation ==========
    print("\n--- Tracking Metrics (MOTA/MOTP/IDF1/ID Switches) ---")

    try:
        summary = mot_metrics(pred_mota_file, gt_mota_file, scale=0.01)

        # Print summary
        print("\nMOT Metrics Summary:")
        print(summary)

        # Extract key metrics
        overall = summary.loc['OVERALL']

        metrics['MOTA'] = overall['mota'] * 100  # Convert to percentage
        metrics['MOTP'] = overall['motp']
        metrics['IDF1'] = overall['idf1'] * 100  # Convert to percentage
        metrics['num_switches'] = overall['num_switches']
        metrics['num_fragmentations'] = overall.get('num_fragmentations', 0)
        metrics['num_false_positives'] = overall.get('num_false_positives', 0)
        metrics['num_misses'] = overall.get('num_misses', 0)

        print(f"\nKey Metrics:")
        print(f"  MOTA:           {metrics['MOTA']:.2f}%")
        print(f"  MOTP:           {metrics['MOTP']:.4f}")
        print(f"  IDF1:           {metrics['IDF1']:.2f}%")
        print(f"  ID Switches:    {int(metrics['num_switches'])}")
        print(f"  Fragmentations: {int(metrics['num_fragmentations'])}")
        print(f"  False Positives: {int(metrics['num_false_positives'])}")
        print(f"  Misses:         {int(metrics['num_misses'])}")

    except Exception as e:
        print(f"Error computing MOTA/MOTP: {e}")
        import traceback
        traceback.print_exc()
        metrics['MOTA'] = 0
        metrics['MOTP'] = 0
        metrics['num_switches'] = 0

    return metrics

def save_results(tracklets, cluster_labels, output_file):
    """Save clustering results to file"""
    print(f"\nSaving results to {output_file}...")

    with open(output_file, 'w') as f:
        f.write("tracklet_idx,cam_id,track_id,start_frame,end_frame,length,cluster_id,gt_id\n")

        for i, tracklet in enumerate(tracklets):
            f.write(f"{i},{tracklet.cam_id + 1},{tracklet.track_id},"
                    f"{tracklet.start_frame},{tracklet.end_frame},{tracklet.length},"
                    f"{cluster_labels[i]},{tracklet.gt_id}\n")

    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Multi-camera tracklet association with full evaluation')
    parser.add_argument('--dataset', type=str,
                        default='/path/to/dataset',
                        help='Path to dataset directory')
    parser.add_argument('--mot_files', type=str, nargs='+',
                        default=[
                            'BoTSORT_results/mmcows_test_cam1.txt',
                            'BoTSORT_results/mmcows_test_cam2.txt',
                            'BoTSORT_results/mmcows_test_cam3.txt',
                            'BoTSORT_results/mmcows_test_cam4.txt'
                        ],
                        help='Paths to MOT format tracking files for each camera')
    parser.add_argument('--gt_moda', type=str,
                        default='/path/to/dataset/sequencename/gt_moda.txt',
                        help='Path to ground truth MODA file')
    parser.add_argument('--gt_mota', type=str,
                        default='/path/to/dataset/sequencename/gt_mota.txt',
                        help='Path to ground truth MOTA file')
    parser.add_argument('--n_clusters', type=int, default=21,
                        help='Number of clusters (number of cows)')
    parser.add_argument('--spatial_threshold', type=float, default=200,
                        help='Maximum spatial distance for connection (cm)')
    parser.add_argument('--temporal_threshold', type=int, default=30,
                        help='Minimum time gap for same-camera tracklets (frames)')
    parser.add_argument('--fill_gaps', action='store_true',
                        help='Fill small gaps in tracks using interpolation')
    parser.add_argument('--max_gap', type=int, default=30,
                        help='Maximum gap size to fill (frames)')
    parser.add_argument('--output_dir', type=str, default='spectral_clustering_results',
                        help='Output directory for results')
    parser.add_argument('--seq_num', type=int, default=1,
                        help='Sequence number for MOTA format')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Multi-Camera Tracklet Association with Full Evaluation")
    print("=" * 80)

    # Step 1: Load tracklets from MOT files
    print("\nStep 1: Loading tracklets from MOT files...")
    all_tracklets = []

    for cam_id, mot_file in enumerate(args.mot_files):
        tracklets = read_mot_file(mot_file, cam_id)
        all_tracklets.extend(tracklets)

    print(f"\nTotal tracklets loaded: {len(all_tracklets)}")

    # ===== CRITICAL: Create frame mapping =====
    print("\n***** Creating frame mapping *****")
    frame_mapping, reverse_mapping = create_frame_mapping(args.mot_files)

    # Save frame mapping for reference
    mapping_file = os.path.join(args.output_dir, 'frame_mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("original_frame,sequential_frame\n")
        for orig, seq in sorted(frame_mapping.items()):
            f.write(f"{orig},{seq}\n")
    print(f"Frame mapping saved to {mapping_file}")

    # Step 2: Project tracklets to 3D world coordinates
    print("\nStep 2: Projecting tracklets to 3D world coordinates...")
    project_tracklets_to_3d(all_tracklets, args.dataset)

    # Step 3: Load ground truth IDs (for IDF1)
    print("\nStep 3: Loading ground truth IDs...")
    load_ground_truth(args.dataset, all_tracklets)

    # Step 4: Build similarity matrix (Xu's Approach)
    print("\nStep 4: Building similarity matrix (Second Script's Approach)...")
    similarity_matrix = build_similarity_matrix(
        all_tracklets,
        spatial_threshold=args.spatial_threshold,
        temporal_threshold=args.temporal_threshold
    )

    # Step 5: Perform spectral clustering using sklearn
    print("\nStep 5: Performing spectral clustering (sklearn implementation)...")
    cluster_labels = spectral_clustering_sklearn(
        similarity_matrix,
        n_clusters=args.n_clusters
    )

    # Step 6: Generate global tracks from clusters
    print("\nStep 6: Generating global tracks from clusters...")
    global_tracks = generate_global_tracks(all_tracklets, cluster_labels)

    # Optional: Fill gaps
    if args.fill_gaps:
        print(f"\nFilling gaps (max gap: {args.max_gap} frames)...")
        global_tracks = fill_track_gaps(global_tracks, max_gap=args.max_gap)

    # Step 7: Save prediction files WITH FRAME MAPPING
    print("\nStep 7: Saving prediction files...")

    pred_moda_file = os.path.join(args.output_dir, 'pred_moda.txt')
    pred_mota_file = os.path.join(args.output_dir, 'pred_mota.txt')
    cluster_file = os.path.join(args.output_dir, 'clustering_results.csv')

    # CRITICAL: Pass frame_mapping to save functions
    save_moda_format(global_tracks, pred_moda_file, frame_mapping=frame_mapping)
    save_mota_format(global_tracks, pred_mota_file, seq_num=args.seq_num, frame_mapping=frame_mapping)
    save_results(all_tracklets, cluster_labels, cluster_file)

    # ========== NEW: COMPREHENSIVE VERIFICATION ==========
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION VERIFICATION")
    print("=" * 80)

    # 1. Detailed coordinate verification
    matched_df = verify_coordinate_system(pred_moda_file, args.gt_moda)

    # 2. Visualize sample frames
    for frame_num in [1, 50, 100, 150]:
        try:
            visualize_matches_sample_frame(pred_moda_file, args.gt_moda,
                                           frame_num=frame_num,
                                           output_dir=args.output_dir)
        except:
            pass

    # 3. Threshold sensitivity analysis
    analyze_threshold_sensitivity(pred_moda_file, args.gt_moda)

    # 4. Cluster quality analysis
    check_cluster_quality(all_tracklets, cluster_labels, args.output_dir)

    # 5. Verify frame alignment (keep existing)
    verify_frame_alignment(pred_moda_file, args.gt_moda, pred_mota_file, args.gt_mota)

    # Step 8: Comprehensive evaluation
    print("\nStep 8: Running comprehensive evaluation...")

    metrics = evaluate_tracking(
        pred_moda_file=pred_moda_file,
        pred_mota_file=pred_mota_file,
        gt_moda_file=args.gt_moda,
        gt_mota_file=args.gt_mota
    )

    # Step 9: Compute IDF1 (from tracklet-level matching)
    print("\nStep 9: Computing tracklet-level IDF1...")
    idf1_tracklet = compute_idf1(all_tracklets, cluster_labels)

    # Step 10: Save metrics summary
    print("\nStep 10: Saving metrics summary...")

    metrics_file = os.path.join(args.output_dir, 'metrics_summary.txt')
    with open(metrics_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE TRACKING METRICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("Detection Metrics:\n")
        f.write(f"  MODA:      {metrics.get('MODA', 0):.2f}%\n")
        f.write(f"  MODP:      {metrics.get('MODP', 0):.2f}%\n")
        f.write(f"  Recall:    {metrics.get('recall', 0):.2f}%\n")
        f.write(f"  Precision: {metrics.get('precision', 0):.2f}%\n\n")

        f.write("Tracking Metrics:\n")
        f.write(f"  MOTA:           {metrics.get('MOTA', 0):.2f}%\n")
        f.write(f"  MOTP:           {metrics.get('MOTP', 0):.4f}\n")
        f.write(f"  IDF1 (MOT):     {metrics.get('IDF1', 0):.2f}%\n")
        f.write(f"  IDF1 (Tracklet):{idf1_tracklet * 100:.2f}%\n")
        f.write(f"  ID Switches:    {int(metrics.get('num_switches', 0))}\n")
        f.write(f"  Fragmentations: {int(metrics.get('num_fragmentations', 0))}\n")
        f.write(f"  False Positives:{int(metrics.get('num_false_positives', 0))}\n")
        f.write(f"  Misses:         {int(metrics.get('num_misses', 0))}\n\n")

        f.write("Clustering Statistics:\n")
        f.write(f"  Number of clusters: {args.n_clusters}\n")
        f.write(f"  Number of tracklets: {len(all_tracklets)}\n")
        f.write(f"  Spatial threshold: {args.spatial_threshold} cm\n")
        f.write(f"  Temporal threshold: {args.temporal_threshold} frames\n")

    print(f"\nMetrics saved to {metrics_file}")

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nDetection Metrics:")
    print(f"  MODA:      {metrics.get('MODA', 0):.2f}%")
    print(f"  MODP:      {metrics.get('MODP', 0):.2f}%")
    print(f"\nTracking Metrics:")
    print(f"  MOTA:      {metrics.get('MOTA', 0):.2f}%")
    print(f"  MOTP:      {metrics.get('MOTP', 0):.4f}")
    print(f"  IDF1:      {metrics.get('IDF1', 0):.2f}%")
    print(f"  ID Switches: {int(metrics.get('num_switches', 0))}")
    print("=" * 80)

    # Print per-cluster statistics
    print("\n" + "=" * 80)
    print("PER-CLUSTER STATISTICS")
    print("=" * 80)

    for cluster_id in sorted(global_tracks.keys()):
        cluster_tracklets = [t for i, t in enumerate(all_tracklets)
                             if cluster_labels[i] == cluster_id]

        # Find majority ground truth ID
        gt_counts = defaultdict(int)
        for tracklet in cluster_tracklets:
            if tracklet.gt_id is not None:
                gt_counts[tracklet.gt_id] += tracklet.length

        majority_gt = max(gt_counts, key=gt_counts.get) if gt_counts else None
        purity = gt_counts[majority_gt] / sum(gt_counts.values()) if gt_counts else 0

        # Camera distribution
        cam_counts = defaultdict(int)
        for tracklet in cluster_tracklets:
            cam_counts[tracklet.cam_id + 1] += 1

        # Temporal span
        num_frames = len(global_tracks[cluster_id])

        print(f"\nCluster {cluster_id}:")
        print(f"  Tracklets: {len(cluster_tracklets)}")
        print(f"  Frames: {num_frames}")
        print(f"  GT ID: {majority_gt} (purity: {purity:.2%})")
        print(f"  Cameras: {dict(cam_counts)}")

    print("\n" + "=" * 80)
    print("All results saved to:", args.output_dir)
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
