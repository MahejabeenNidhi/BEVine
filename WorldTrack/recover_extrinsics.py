# WorldTrack/recover_extrinsics.py
"""
Recover correct camera extrinsics for MmCows dataset.

The original calibration files have near-zero tvec, making projections useless.
This script recovers correct [R|t] by solving the ground-plane homography
using known 3D world positions and their 2D bounding box foot-points.

Usage:
    python recover_extrinsics.py \
        --data_dir /path/to/mmcows_050326_train \
        --output_dir debug_calibration

Output:
    calibrations/extrinsic_recovered/C{1-4}_extrinsic.npz
    debug_calibration/vis_*.png
    debug_calibration/recovery_report.txt
"""

import os
import sys
import json
import ast
import argparse
import numpy as np
import cv2
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════
#  Constants (must match mmcows_dataset.py)
# ═══════════════════════════════════════════════════════════════════════════

NUM_CAM = 4
CAM_NAMES = ['C1', 'C2', 'C3', 'C4']
IMG_SHAPE = (2800, 4480)  # H, W
GRID_HEIGHT = 117
GRID_WIDTH = 192
GRID_CELL_SIZE = 10  # cm
X_MIN_CM = -879
Y_MIN_CM = -646

# World coordinate transform: world = M @ [grid_x, grid_y, 1]
M_WORLD = np.array([
    [GRID_CELL_SIZE, 0, X_MIN_CM],
    [0, GRID_CELL_SIZE, Y_MIN_CM],
    [0, 0, 1]
], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
#  Report Logger
# ═══════════════════════════════════════════════════════════════════════════

class Report:
    def __init__(self):
        self.lines = []
        self.warnings = []
        self.errors = []

    def section(self, title):
        bar = "=" * 70
        self.lines.extend([f"\n{bar}", f"  {title}", bar])
        print(f"\n{bar}\n  {title}\n{bar}")

    def log(self, msg, indent=0):
        line = "  " * indent + msg
        self.lines.append(line)
        print(line)

    def warn(self, msg):
        line = f"  ⚠ WARNING: {msg}"
        self.lines.append(line)
        self.warnings.append(msg)
        print(line)

    def error(self, msg):
        line = f"  ✗ ERROR: {msg}"
        self.lines.append(line)
        self.errors.append(msg)
        print(line)

    def ok(self, msg):
        line = f"  ✓ {msg}"
        self.lines.append(line)
        print(line)

    def save(self, path):
        with open(path, 'w') as f:
            f.write('\n'.join(self.lines))
            f.write(f"\n\n{'='*70}\n")
            f.write(f"  SUMMARY: {len(self.warnings)} warnings, {len(self.errors)} errors\n")
            for w in self.warnings:
                f.write(f"    ⚠ {w}\n")
            for e in self.errors:
                f.write(f"    ✗ {e}\n")
        print(f"\nReport saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def get_worldgrid_from_pos(pos):
    gy = pos % GRID_HEIGHT
    gx = pos // GRID_HEIGHT
    return gx, gy


def load_intrinsics(data_dir):
    intrinsics = []
    for cam_name in CAM_NAMES:
        path = os.path.join(data_dir, 'calibrations', 'intrinsic_zero',
                            f'{cam_name}_intrinsic.txt')
        with open(path, 'r') as f:
            data = ast.literal_eval(f.read())
        K = np.array(data['camera_matrix'], dtype=np.float64)
        intrinsics.append(K)
    return intrinsics


def load_original_rvec_tvec(data_dir):
    """Load rvec/tvec from original extrinsic files for comparison."""
    results = []
    for cam_name in CAM_NAMES:
        path = os.path.join(data_dir, 'calibrations', 'extrinsic',
                            f'{cam_name}_extrinsic.txt')
        with open(path, 'r') as f:
            lines = f.readlines()

        rvec_vals, tvec_vals = [], []
        reading = None
        for line in lines:
            s = line.strip()
            if 'Rotation Vector' in s:
                reading = 'rvec'; continue
            elif 'Translation Vector' in s:
                reading = 'tvec'; continue
            elif 'Rotation Matrix' in s:
                reading = None; continue
            if not s:
                continue
            try:
                if reading == 'rvec' and len(rvec_vals) < 3:
                    rvec_vals.append(float(s))
                elif reading == 'tvec' and len(tvec_vals) < 3:
                    tvec_vals.append(float(s))
            except ValueError:
                pass

        rvec = np.array(rvec_vals, dtype=np.float64)
        tvec = np.array(tvec_vals, dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        results.append({'rvec': rvec, 'tvec': tvec, 'R': R})
    return results


def collect_correspondences(data_dir, report):
    """
    Collect 3D→2D correspondences from ALL annotation frames.

    3D: world position on ground plane (from positionID)
    2D: bbox foot-point = bottom-center of bounding box
    """
    report.section("Collecting 3D↔2D Correspondences")

    ann_dir = os.path.join(data_dir, 'annotations_positions')
    ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.json')])
    report.log(f"Annotation frames: {len(ann_files)}")

    corr = {cam: {'pts_3d': [], 'pts_2d': [], 'pids': [], 'frames': []}
            for cam in range(NUM_CAM)}

    total_annotations = 0
    for ann_file in ann_files:
        frame_num = int(ann_file.split('.')[0])
        with open(os.path.join(ann_dir, ann_file)) as f:
            anns = json.load(f)

        for ped in anns:
            gx, gy = get_worldgrid_from_pos(ped['positionID'])
            world = M_WORLD @ np.array([gx, gy, 1.0])
            pt_3d = np.array([world[0], world[1], 0.0])

            for cam_idx in range(NUM_CAM):
                view = ped['views'][cam_idx]
                if view['xmin'] == -1:
                    continue

                # Foot point = bottom center of bbox
                foot_x = (view['xmin'] + view['xmax']) / 2.0
                foot_y = view['ymax']

                # Sanity check: foot point should be within image
                if not (0 <= foot_x <= IMG_SHAPE[1] and 0 <= foot_y <= IMG_SHAPE[0]):
                    continue

                corr[cam_idx]['pts_3d'].append(pt_3d)
                corr[cam_idx]['pts_2d'].append([foot_x, foot_y])
                corr[cam_idx]['pids'].append(ped['personID'])
                corr[cam_idx]['frames'].append(frame_num)
                total_annotations += 1

    report.log(f"Total correspondences collected: {total_annotations}")
    for cam_idx in range(NUM_CAM):
        n = len(corr[cam_idx]['pts_3d'])
        report.log(f"  {CAM_NAMES[cam_idx]}: {n} correspondences", 1)
        if n < 6:
            report.warn(f"{CAM_NAMES[cam_idx]}: Only {n} correspondences — may be insufficient")

    return corr


# ═══════════════════════════════════════════════════════════════════════════
#  Extrinsic Recovery Methods
# ═══════════════════════════════════════════════════════════════════════════

def recover_via_homography(pts_3d, pts_2d, K, report, cam_name):
    """
    Recover [R|t] from ground-plane correspondences using homography decomposition.

    For Z=0: s·[u,v,1]ᵀ = K·[r₁ r₂ t]·[X,Y,1]ᵀ
    where H = K·[r₁ r₂ t] is the ground-to-image homography.
    """
    pts_3d = np.array(pts_3d, dtype=np.float64)
    pts_2d = np.array(pts_2d, dtype=np.float64)
    pts_world_2d = pts_3d[:, :2]  # only X,Y (Z=0)

    # Find homography with RANSAC
    H, mask = cv2.findHomography(pts_world_2d, pts_2d, cv2.RANSAC,
                                  ransacReprojThreshold=30.0,
                                  maxIters=20000,
                                  confidence=0.999)

    if H is None:
        report.error(f"{cam_name} Homography: findHomography failed")
        return None, None, None, None

    n_inliers = mask.sum() if mask is not None else 0
    report.log(f"  Homography RANSAC: {n_inliers}/{len(pts_3d)} inliers", 1)

    # Decompose: K⁻¹·H = λ·[r₁ r₂ t]
    K_inv = np.linalg.inv(K)
    M = K_inv @ H

    # Scale factor: ‖r₁‖ should be 1
    lam1 = np.linalg.norm(M[:, 0])
    lam2 = np.linalg.norm(M[:, 1])
    lam = (lam1 + lam2) / 2.0

    if lam < 1e-10:
        report.error(f"{cam_name} Homography: degenerate scale λ={lam:.2e}")
        return None, None, None, None

    r1 = M[:, 0] / lam
    r2 = M[:, 1] / lam
    t = M[:, 2] / lam
    r3 = np.cross(r1, r2)

    # Orthogonalize R via SVD
    R_approx = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt

    # Ensure det(R) = +1
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = U @ Vt

    # Check points are in front of camera
    n_behind = 0
    for pt in pts_3d[:10]:
        cam_pt = R @ pt + t
        if cam_pt[2] < 0:
            n_behind += 1

    if n_behind > 5:
        # Flip solution (other homography sign)
        report.log(f"  Flipping homography sign ({n_behind}/10 points behind camera)", 1)
        r1 = -M[:, 0] / lam
        r2 = -M[:, 1] / lam
        t = -M[:, 2] / lam
        r3 = np.cross(r1, r2)
        R_approx = np.column_stack([r1, r2, r3])
        U, _, Vt = np.linalg.svd(R_approx)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = U @ Vt

    # Compute reprojection errors
    errors = compute_reprojection_errors(pts_3d, pts_2d, R, t, K)

    return R, t, errors, mask.flatten().astype(bool) if mask is not None else None


def recover_via_pnp(pts_3d, pts_2d, K, report, cam_name):
    """Recover [R|t] using solvePnPRansac."""
    pts_3d = np.array(pts_3d, dtype=np.float64)
    pts_2d = np.array(pts_2d, dtype=np.float64)

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d.reshape(-1, 1, 3),
            pts_2d.reshape(-1, 1, 2),
            K, distCoeffs=np.zeros(5),
            iterationsCount=20000,
            reprojectionError=30.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    except cv2.error as e:
        report.error(f"{cam_name} PnP: {e}")
        return None, None, None, None

    if not success:
        report.error(f"{cam_name} PnP: solvePnPRansac failed")
        return None, None, None, None

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()

    n_inliers = len(inliers) if inliers is not None else 0
    report.log(f"  PnP RANSAC: {n_inliers}/{len(pts_3d)} inliers", 1)

    errors = compute_reprojection_errors(pts_3d, pts_2d, R, t, K)

    mask = np.zeros(len(pts_3d), dtype=bool)
    if inliers is not None:
        mask[inliers.flatten()] = True

    return R, t, errors, mask


def compute_reprojection_errors(pts_3d, pts_2d, R, t, K):
    """Compute per-point reprojection errors."""
    errors = []
    for pt3d, pt2d in zip(pts_3d, pts_2d):
        cam = R @ pt3d + t
        if cam[2] <= 0:
            errors.append(np.inf)
            continue
        pix = K @ cam
        proj = np.array([pix[0] / pix[2], pix[1] / pix[2]])
        errors.append(np.linalg.norm(proj - pt2d))
    return np.array(errors)


# ═══════════════════════════════════════════════════════════════════════════
#  Main Recovery Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def recover_all_cameras(data_dir, report):
    """Recover extrinsics for all cameras."""
    report.section("Loading Calibration Data")

    intrinsics = load_intrinsics(data_dir)
    orig = load_original_rvec_tvec(data_dir)

    for cam_idx in range(NUM_CAM):
        K = intrinsics[cam_idx]
        report.log(f"{CAM_NAMES[cam_idx]}: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, "
                   f"cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
        report.log(f"  Original |tvec|={np.linalg.norm(orig[cam_idx]['tvec']):.8f}", 1)

    corr = collect_correspondences(data_dir, report)

    report.section("Recovering Extrinsics")
    recovered = {}

    for cam_idx in range(NUM_CAM):
        report.log(f"\n{'─'*50}")
        report.log(f"Camera {CAM_NAMES[cam_idx]} ({len(corr[cam_idx]['pts_3d'])} correspondences)")
        report.log(f"{'─'*50}")

        K = intrinsics[cam_idx]
        pts_3d = corr[cam_idx]['pts_3d']
        pts_2d = corr[cam_idx]['pts_2d']

        if len(pts_3d) < 6:
            report.error(f"{CAM_NAMES[cam_idx]}: Insufficient correspondences ({len(pts_3d)} < 6)")
            continue

        # ── Method 1: Homography ──
        R_h, t_h, err_h, mask_h = recover_via_homography(
            pts_3d, pts_2d, K, report, CAM_NAMES[cam_idx])

        # ── Method 2: PnP ──
        R_p, t_p, err_p, mask_p = recover_via_pnp(
            pts_3d, pts_2d, K, report, CAM_NAMES[cam_idx])

        # ── Compare and select best ──
        results = []
        if R_h is not None:
            finite_h = err_h[np.isfinite(err_h)]
            if len(finite_h) > 0:
                results.append(('Homography', R_h, t_h, err_h, np.mean(finite_h),
                                np.median(finite_h)))
                report.log(f"  Homography: mean={np.mean(finite_h):.1f}px, "
                           f"median={np.median(finite_h):.1f}px, "
                           f"max={np.max(finite_h):.1f}px")

        if R_p is not None:
            finite_p = err_p[np.isfinite(err_p)]
            if len(finite_p) > 0:
                results.append(('PnP', R_p, t_p, err_p, np.mean(finite_p),
                                np.median(finite_p)))
                report.log(f"  PnP:        mean={np.mean(finite_p):.1f}px, "
                           f"median={np.median(finite_p):.1f}px, "
                           f"max={np.max(finite_p):.1f}px")

        if not results:
            report.error(f"{CAM_NAMES[cam_idx]}: All recovery methods failed!")
            continue

        # Select method with lowest median error
        results.sort(key=lambda x: x[5])  # sort by median error
        best_name, best_R, best_t, best_err, best_mean, best_med = results[0]

        cam_center = -best_R.T @ best_t
        report.ok(f"Selected: {best_name} (median={best_med:.1f}px)")
        report.log(f"  det(R) = {np.linalg.det(best_R):.8f}")
        report.log(f"  t = [{best_t[0]:.4f}, {best_t[1]:.4f}, {best_t[2]:.4f}]")
        report.log(f"  Camera center (world cm): "
                   f"({cam_center[0]:.1f}, {cam_center[1]:.1f}, {cam_center[2]:.1f})")

        # Sanity checks
        if best_med > 100:
            report.warn(f"{CAM_NAMES[cam_idx]}: Median error {best_med:.1f}px is high — "
                        f"check bbox annotation quality")
        elif best_med > 50:
            report.warn(f"{CAM_NAMES[cam_idx]}: Median error {best_med:.1f}px — "
                        f"acceptable for bbox-based recovery")
        else:
            report.ok(f"{CAM_NAMES[cam_idx]}: Median error {best_med:.1f}px — excellent!")

        # Check camera is above the scene (Z should be positive for overhead cameras)
        if cam_center[2] < 0:
            report.warn(f"{CAM_NAMES[cam_idx]}: Camera Z={cam_center[2]:.1f} is below ground. "
                        f"Check if Z convention is correct.")

        # Store
        Rt = np.hstack([best_R, best_t.reshape(3, 1)])
        rvec, _ = cv2.Rodrigues(best_R)
        recovered[cam_idx] = {
            'R': best_R,
            't': best_t,
            'rvec': rvec.flatten(),
            'Rt': Rt,
            'cam_center': cam_center,
            'method': best_name,
            'mean_error': best_mean,
            'median_error': best_med,
            'errors': best_err,
        }

    return recovered, intrinsics, corr, orig


# ═══════════════════════════════════════════════════════════════════════════
#  Visualizations
# ═══════════════════════════════════════════════════════════════════════════

def vis_before_after(recovered, intrinsics, orig, data_dir, output_dir, report):
    """Side-by-side: original (broken) vs recovered projections."""
    report.section("VIS: Before/After Projection Overlay")

    ann_dir = os.path.join(data_dir, 'annotations_positions')
    ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.json')])

    # Use first frame
    frame_num = int(ann_files[0].split('.')[0])
    with open(os.path.join(ann_dir, ann_files[0])) as f:
        anns = json.load(f)

    # Load images
    images = []
    for cam_name in CAM_NAMES:
        img_dir = os.path.join(data_dir, 'Image_subsets', cam_name)
        for ext in ['.png', '.jpg']:
            img_path = os.path.join(img_dir, f'{frame_num:08d}{ext}')
            if os.path.exists(img_path):
                images.append(np.array(Image.open(img_path).convert('RGB')))
                break
        else:
            images.append(np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], 3), dtype=np.uint8))

    fig, axes = plt.subplots(NUM_CAM, 2, figsize=(28, 7 * NUM_CAM))

    for cam_idx in range(NUM_CAM):
        K = intrinsics[cam_idx]

        for col, (label, Rt) in enumerate([
            ("ORIGINAL (broken tvec≈0)",
             np.hstack([orig[cam_idx]['R'], orig[cam_idx]['tvec'].reshape(3, 1)])),
            ("RECOVERED",
             recovered[cam_idx]['Rt'] if cam_idx in recovered else None),
        ]):
            ax = axes[cam_idx, col]
            ax.imshow(images[cam_idx])

            if Rt is None:
                ax.set_title(f'{CAM_NAMES[cam_idx]} — {label}\n(FAILED)',
                             fontsize=13, color='red')
                ax.axis('off')
                continue

            n_projected = 0
            n_in_image = 0

            for ped in anns:
                gx, gy = get_worldgrid_from_pos(ped['positionID'])
                world = M_WORLD @ np.array([gx, gy, 1.0])

                # Draw annotated bbox
                view = ped['views'][cam_idx]
                if view['xmin'] != -1:
                    rect = patches.Rectangle(
                        (view['xmin'], view['ymin']),
                        view['xmax'] - view['xmin'],
                        view['ymax'] - view['ymin'],
                        lw=2, ec='lime', fc='none', ls='--'
                    )
                    ax.add_patch(rect)

                    # Mark foot point
                    foot_x = (view['xmin'] + view['xmax']) / 2.0
                    foot_y = view['ymax']
                    ax.plot(foot_x, foot_y, 'g+', markersize=12, markeredgewidth=2)

                # Project world point
                pt_h = np.array([world[0], world[1], 0.0, 1.0])
                cam_pt = Rt @ pt_h
                if cam_pt[2] <= 0:
                    continue
                pix = K @ cam_pt
                px, py = pix[0] / pix[2], pix[1] / pix[2]
                n_projected += 1

                H, W = IMG_SHAPE
                if 0 <= px < W and 0 <= py < H:
                    n_in_image += 1
                    ax.plot(px, py, 'ro', markersize=10, markeredgecolor='white',
                            markeredgewidth=1.5, zorder=5)
                    ax.annotate(f'{ped["personID"]}', (px, py - 20),
                                color='red', fontsize=9, fontweight='bold', ha='center',
                                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.8))

            ax.set_xlim(0, IMG_SHAPE[1])
            ax.set_ylim(IMG_SHAPE[0], 0)
            ax.set_title(f'{CAM_NAMES[cam_idx]} — {label}\n'
                         f'{n_in_image}/{n_projected} projected in image',
                         fontsize=13, fontweight='bold')
            ax.axis('off')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='Projected ground point'),
        Line2D([0], [0], marker='+', color='lime', markersize=10,
               markeredgewidth=2, label='Bbox foot point (GT)'),
        Line2D([0], [0], linestyle='--', color='lime', lw=2, label='Bbox (GT)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=12, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(f'Extrinsic Recovery: Before vs After — Frame {frame_num}\n'
                 f'Red circles should land on green crosses (bbox foot)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    path = os.path.join(output_dir, 'vis_before_after.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    report.log(f"Saved: {path}")


def vis_bev_cameras(recovered, data_dir, output_dir, report):
    """BEV plot with recovered camera positions."""
    report.section("VIS: BEV with Recovered Camera Positions")

    ann_dir = os.path.join(data_dir, 'annotations_positions')
    ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.json')])
    with open(os.path.join(ann_dir, ann_files[0])) as f:
        anns = json.load(f)

    fig, (ax_w, ax_g) = plt.subplots(1, 2, figsize=(22, 9))
    cam_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']

    M_inv = np.linalg.inv(M_WORLD)

    for ax, is_grid in [(ax_w, False), (ax_g, True)]:
        # Subjects
        for ped in anns:
            gx, gy = get_worldgrid_from_pos(ped['positionID'])
            world = M_WORLD @ np.array([gx, gy, 1.0])
            x, y = (gx, gy) if is_grid else (world[0], world[1])
            ax.plot(x, y, 'ko', markersize=8, zorder=3)
            off = 1 if is_grid else 10
            ax.annotate(f'{ped["personID"]}', (x + off, y + off), fontsize=8)

        # Cameras
        for cam_idx in range(NUM_CAM):
            if cam_idx not in recovered:
                continue
            cc = recovered[cam_idx]['cam_center']
            if is_grid:
                gc = M_inv @ np.array([cc[0], cc[1], 1.0])
                x, y = gc[0], gc[1]
            else:
                x, y = cc[0], cc[1]

            ax.plot(x, y, '^', color=cam_colors[cam_idx], markersize=16,
                    markeredgecolor='black', markeredgewidth=2, zorder=10)
            label = (f'{CAM_NAMES[cam_idx]}\n'
                     f'({cc[0]:.0f},{cc[1]:.0f},z={cc[2]:.0f})')
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 15),
                        ha='center', fontsize=10, fontweight='bold',
                        color=cam_colors[cam_idx],
                        bbox=dict(boxstyle='round', fc='white', alpha=0.9))

        if is_grid:
            rect = patches.Rectangle((0, 0), GRID_WIDTH, GRID_HEIGHT,
                                     lw=2, ec='black', fc='none')
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')
            ax.set_title('Grid Coordinates', fontsize=14)
        else:
            rect = patches.Rectangle((X_MIN_CM, Y_MIN_CM),
                                     GRID_WIDTH * GRID_CELL_SIZE,
                                     GRID_HEIGHT * GRID_CELL_SIZE,
                                     lw=2, ec='black', fc='none')
            ax.set_xlabel('World X (cm)')
            ax.set_ylabel('World Y (cm)')
            ax.set_title('World Coordinates (cm)', fontsize=14)

        ax.add_patch(rect)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('BEV with Recovered Camera Positions\n'
                 'Cameras should be at distinct locations around the scene perimeter',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'vis_bev_cameras.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    report.log(f"Saved: {path}")


def vis_forward_geometry(recovered, intrinsics, output_dir, report):
    """Verify BEV grid → camera pixel coverage with recovered extrinsics."""
    report.section("VIS: Forward Geometry (BEV → Camera)")

    # Sample BEV grid
    step = 4
    gx = np.arange(0, GRID_WIDTH, step, dtype=np.float64)
    gy = np.arange(0, GRID_HEIGHT, step, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(gy, gx, indexing='ij')
    bev_grid = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    n_pts = len(bev_grid)

    # Convert to world 3D
    bev_3d = np.zeros((n_pts, 4))
    for i, (gxi, gyi) in enumerate(bev_grid):
        w = M_WORLD @ np.array([gxi, gyi, 1.0])
        bev_3d[i] = [w[0], w[1], 0.0, 1.0]

    fig, axes = plt.subplots(1, NUM_CAM + 1, figsize=(5 * (NUM_CAM + 1), 5))

    axes[0].scatter(bev_grid[:, 0], bev_grid[:, 1], s=2, c='blue', alpha=0.5)
    axes[0].set_xlim(0, GRID_WIDTH)
    axes[0].set_ylim(0, GRID_HEIGHT)
    axes[0].set_aspect('equal')
    axes[0].set_title(f'BEV Grid ({n_pts} pts)')

    for cam_idx in range(NUM_CAM):
        ax = axes[cam_idx + 1]

        if cam_idx not in recovered:
            ax.set_title(f'{CAM_NAMES[cam_idx]}: FAILED', color='red')
            continue

        Rt = recovered[cam_idx]['Rt']
        K = intrinsics[cam_idx]

        vis_x, vis_y = [], []
        for pt in bev_3d:
            cam = Rt @ pt
            if cam[2] <= 0:
                continue
            pix = K @ cam
            px, py = pix[0] / pix[2], pix[1] / pix[2]
            if 0 <= px < IMG_SHAPE[1] and 0 <= py < IMG_SHAPE[0]:
                vis_x.append(px)
                vis_y.append(py)

        ax.scatter(vis_x, vis_y, s=1, c='blue', alpha=0.3)
        ax.set_xlim(0, IMG_SHAPE[1])
        ax.set_ylim(IMG_SHAPE[0], 0)
        ax.set_aspect('equal')
        pct = len(vis_x) / n_pts * 100
        ax.set_title(f'{CAM_NAMES[cam_idx]}: {len(vis_x)}/{n_pts} ({pct:.0f}%)')

        if pct < 30:
            report.warn(f"{CAM_NAMES[cam_idx]}: Only {pct:.0f}% of BEV visible — "
                        f"camera has limited coverage")

        report.log(f"  {CAM_NAMES[cam_idx]}: {len(vis_x)}/{n_pts} ({pct:.0f}%) "
                   f"BEV cells visible")

    plt.suptitle('BEV → Camera Projection (Recovered Extrinsics)\n'
                 'Each camera should see a significant portion of the BEV grid',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'vis_forward_geometry.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    report.log(f"Saved: {path}")


def vis_error_distribution(recovered, output_dir, report):
    """Reprojection error distribution per camera."""
    fig, axes = plt.subplots(1, NUM_CAM, figsize=(5 * NUM_CAM, 4))

    for cam_idx in range(NUM_CAM):
        ax = axes[cam_idx]
        if cam_idx not in recovered:
            ax.set_title(f'{CAM_NAMES[cam_idx]}: N/A')
            continue

        errs = recovered[cam_idx]['errors']
        finite = errs[np.isfinite(errs)]
        ax.hist(finite, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.median(finite), color='red', linestyle='--', linewidth=2,
                   label=f'Median={np.median(finite):.1f}px')
        ax.axvline(np.mean(finite), color='orange', linestyle='--', linewidth=2,
                   label=f'Mean={np.mean(finite):.1f}px')
        ax.set_xlabel('Reprojection Error (px)')
        ax.set_ylabel('Count')
        ax.set_title(f'{CAM_NAMES[cam_idx]} ({recovered[cam_idx]["method"]})')
        ax.legend(fontsize=8)

    plt.suptitle('Reprojection Error Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'vis_error_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    report.log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Save Recovered Extrinsics
# ═══════════════════════════════════════════════════════════════════════════

def save_recovered(recovered, data_dir, report):
    """Save recovered extrinsics as .npz files."""
    report.section("Saving Recovered Extrinsics")

    save_dir = os.path.join(data_dir, 'calibrations', 'extrinsic_recovered')
    os.makedirs(save_dir, exist_ok=True)

    for cam_idx in range(NUM_CAM):
        if cam_idx not in recovered:
            report.error(f"{CAM_NAMES[cam_idx]}: No recovered extrinsic to save")
            continue

        r = recovered[cam_idx]
        path = os.path.join(save_dir, f'{CAM_NAMES[cam_idx]}_extrinsic.npz')
        np.savez(path,
                 R=r['R'].astype(np.float64),
                 t=r['t'].astype(np.float64),
                 rvec=r['rvec'].astype(np.float64),
                 Rt=r['Rt'].astype(np.float64),
                 cam_center=r['cam_center'].astype(np.float64),
                 method=str(r['method']),
                 mean_error=float(r['mean_error']),
                 median_error=float(r['median_error']))
        report.ok(f"{CAM_NAMES[cam_idx]}: Saved to {path}")

    report.log(f"\nTo use recovered extrinsics, mmcows_dataset.py will")
    report.log(f"automatically load from: {save_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Recover MmCows camera extrinsics')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to MmCows dataset root')
    parser.add_argument('--output_dir', type=str, default='debug_calibration',
                        help='Output directory for visualizations and report')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report = Report()

    report.section("MmCows Extrinsic Recovery")
    report.log(f"Data: {args.data_dir}")
    report.log(f"Output: {args.output_dir}")

    # ── Recovery ──
    recovered, intrinsics, corr, orig = recover_all_cameras(args.data_dir, report)

    if not recovered:
        report.error("No cameras were successfully recovered!")
        report.save(os.path.join(args.output_dir, 'recovery_report.txt'))
        return

    # ── Save ──
    save_recovered(recovered, args.data_dir, report)

    # ── Visualize ──
    try:
        vis_before_after(recovered, intrinsics, orig, args.data_dir, args.output_dir, report)
    except Exception as e:
        report.error(f"vis_before_after: {e}")
        import traceback; traceback.print_exc()

    try:
        vis_bev_cameras(recovered, args.data_dir, args.output_dir, report)
    except Exception as e:
        report.error(f"vis_bev_cameras: {e}")
        import traceback; traceback.print_exc()

    try:
        vis_forward_geometry(recovered, intrinsics, args.output_dir, report)
    except Exception as e:
        report.error(f"vis_forward_geometry: {e}")
        import traceback; traceback.print_exc()

    try:
        vis_error_distribution(recovered, args.output_dir, report)
    except Exception as e:
        report.error(f"vis_error_distribution: {e}")
        import traceback; traceback.print_exc()

    # ── Final Summary ──
    report.section("FINAL SUMMARY")
    for cam_idx in range(NUM_CAM):
        if cam_idx in recovered:
            r = recovered[cam_idx]
            report.ok(f"{CAM_NAMES[cam_idx]}: {r['method']}, "
                      f"median_err={r['median_error']:.1f}px, "
                      f"center=({r['cam_center'][0]:.0f},{r['cam_center'][1]:.0f},"
                      f"{r['cam_center'][2]:.0f})")
        else:
            report.error(f"{CAM_NAMES[cam_idx]}: FAILED")

    all_ok = all(cam_idx in recovered and recovered[cam_idx]['median_error'] < 100
                 for cam_idx in range(NUM_CAM))

    if all_ok:
        report.ok("All cameras recovered successfully!")
        report.log("\nNext steps:")
        report.log("  1. Inspect vis_before_after.png — red dots should be on green crosses")
        report.log("  2. Inspect vis_bev_cameras.png — cameras at 4 corners of barn")
        report.log("  3. Inspect vis_forward_geometry.png — good image coverage")
        report.log("  4. Run training with recovered extrinsics")
    else:
        report.warn("Some cameras may need manual inspection")

    report.save(os.path.join(args.output_dir, 'recovery_report.txt'))

    print(f"\n{'#'*70}")
    print(f"  Done! Check outputs in: {args.output_dir}/")
    print(f"  Recovered extrinsics: {args.data_dir}/calibrations/extrinsic_recovered/")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()
