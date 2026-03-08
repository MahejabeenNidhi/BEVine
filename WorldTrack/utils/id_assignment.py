# WorldTrack/utils/id_assignment.py
"""
Backward matching: project BEV tracks to camera views via ground-plane
homography, then assign BEV track IDs to per-camera 2D detections using
the Hungarian algorithm.

Includes diagnostic/visualization helpers.
"""

import os
import numpy as np
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
from scipy.optimize import linear_sum_assignment


def compute_ground_plane_homographies(
    intrinsic_original, extrinsic, ref_T_global,
    distortion_coeffs=None,
    verbose=False,
):
    """
    Compute per-camera ground-plane homographies that map ref/grid
    coordinates (Z = 0) to original-resolution image pixels.

    Projection chain:
      ref ─inverse(ref_T_global)─▸ world ─extrinsic─▸ cam
      ─intrinsic─▸ pixel

    For the ground plane (Z = 0 in ref), we drop column 2 of the full
    4×4 projection matrix to obtain a 3×3 homography.

    Parameters
    ----------
    intrinsic_original : (S, 4, 4) tensor
    extrinsic          : (S, 4, 4) tensor
    ref_T_global       : (4, 4) tensor
    distortion_coeffs  : (S, 5) array or None
        Optional per-camera distortion coefficients.  When provided,
        cv2.undistortPoints would be applied to projected pixel coordinates.
        Currently a forward-looking hook; when None (default) behaviour is
        identical to the previous version.
    verbose            : bool

    Returns
    -------
    list of S numpy (3, 3) homography matrices
    """
    import torch

    global_T_ref = torch.inverse(ref_T_global).cpu().float()
    S = intrinsic_original.shape[0]
    homographies = []

    if verbose:
        print(f"\n [HOMOGRAPHY] Computing for {S} cameras")
        print(f" [HOMOGRAPHY] ref_T_global:\n{ref_T_global}")
        print(f" [HOMOGRAPHY] global_T_ref:\n{global_T_ref}")

    for c in range(S):
        K = intrinsic_original[c].cpu().float()
        E = extrinsic[c].cpu().float()

        # Full 4×4 projection: grid → world → cam → pixel
        P = K @ E @ global_T_ref  # 4×4

        # Drop Z-column (col 2) for ground plane Z=0
        H = P[:3, [0, 1, 3]].numpy().copy()  # 3×3

        if verbose:
            print(f"\n [HOMOGRAPHY] Camera {c}:")
            print(f"   K[:3,:3]:\n{K[:3,:3].numpy()}")
            print(f"   E[:3]:\n{E[:3].numpy()}")
            print(f"   H (3x3):\n{H}")

            # Sanity: project grid center
            grid_center = np.array([96.0, 58.0, 1.0])
            p = H @ grid_center
            if abs(p[2]) > 1e-6:
                u, v = p[0] / p[2], p[1] / p[2]
                print(f"   Grid center ({grid_center[0]:.0f},"
                      f"{grid_center[1]:.0f}) → pixel "
                      f"({u:.1f}, {v:.1f})")

        homographies.append(H)

    return homographies


def project_tracks_to_image(bev_positions, homography, img_w, img_h,
                            margin=0.02):
    """
    Project M ground-plane positions through a 3×3 homography.

    Parameters
    ----------
    bev_positions : array-like, shape (M, 2)
    homography    : ndarray (3, 3)
    img_w, img_h  : int
    margin        : float
        Fractional margin beyond image bounds for validity (default 2 %).

    Returns
    -------
    uv    : (M, 2) pixel coordinates
    valid : (M,) boolean mask
    """
    M = len(bev_positions)
    if M == 0:
        return np.empty((0, 2)), np.empty((0,), dtype=bool)

    pts = np.asarray(bev_positions, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((M, 1))
    pts_h = np.hstack([pts, ones])

    projected = (homography @ pts_h.T).T          # M × 3
    w = projected[:, 2]
    safe = np.abs(w) > 1e-6
    uv = np.zeros((M, 2))
    uv[safe] = projected[safe, :2] / w[safe, np.newaxis]

    in_front = w > 0
    in_bounds = (
        (uv[:, 0] >= -img_w * margin) &
        (uv[:, 0] <= img_w * (1 + margin)) &
        (uv[:, 1] >= -img_h * margin) &
        (uv[:, 1] <= img_h * (1 + margin))
    )
    valid = safe & in_front & in_bounds
    return uv, valid


def assign_ids_to_detections(
    bev_track_positions,
    bev_track_ids,
    detections_per_cam,
    homographies,
    img_hw,
    tau_score=0.3,
    tau_match=100.0,
    use_foot_point=True,
    adaptive_tau=True,
    tau_floor=50.0,
    tau_fraction=0.15,
    cost_gate_factor=3.0,
    projection_margin=0.02,
    verbose=False,
):
    """
    For every camera, project BEV tracks to the image plane and run
    Hungarian matching against YOLO 2-D detections.

    Parameters
    ----------
    bev_track_positions : list of (x, y) in ref / grid coords
    bev_track_ids       : list of int track IDs
    detections_per_cam  : dict cam_idx → list of (x1, y1, x2, y2, score)
    homographies        : list of 3 × 3 numpy arrays (per camera)
    img_hw              : (H, W) of the original camera images
    tau_score           : YOLO confidence threshold
    tau_match           : fixed pixel-distance threshold (fallback)
    use_foot_point      : if True, match projection to bbox bottom-centre
    adaptive_tau        : if True, scale threshold with bbox height
    tau_floor           : minimum adaptive threshold (px)
    tau_fraction        : threshold = fraction × bbox height
    cost_gate_factor    : entries > gate × effective_tau set to 1e9
    projection_margin   : fractional margin for projection validity
    verbose             : print diagnostic info

    Returns
    -------
    dict cam_idx → list of (x1, y1, x2, y2, score, track_id)
        track_id == -1 for unmatched detections
    """
    results = {}
    M = len(bev_track_positions)
    img_h, img_w = img_hw

    if verbose:
        ref_label = "foot-pt" if use_foot_point else "centre"
        print(f"\n [ASSIGN] {M} BEV tracks, img_hw=({img_h},{img_w}), "
              f"ref={ref_label}, adaptive_tau={adaptive_tau}")

    for cam_idx, H in enumerate(homographies):
        raw_dets = detections_per_cam.get(cam_idx, [])
        dets = [(x1, y1, x2, y2, s)
                for x1, y1, x2, y2, s in raw_dets if s >= tau_score]

        if verbose:
            print(f"\n [ASSIGN] Cam {cam_idx}: "
                  f"{len(raw_dets)} raw → {len(dets)} after "
                  f"score filter (τ_s={tau_score})")

        if not dets and M == 0:
            results[cam_idx] = []
            continue
        if not dets:
            results[cam_idx] = []
            continue
        if M == 0:
            results[cam_idx] = [
                (x1, y1, x2, y2, s, -1)
                for x1, y1, x2, y2, s in dets
            ]
            continue

        N = len(dets)

        # ── Detection reference points ──────────────
        if use_foot_point:
            det_ref = np.array(
                [((x1 + x2) / 2.0, y2)
                 for x1, y1, x2, y2, _ in dets]
            )
        else:
            det_ref = np.array(
                [((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                 for x1, y1, x2, y2, _ in dets]
            )

        # Also keep centres for diagnostics
        det_centres = np.array(
            [((x1 + x2) / 2.0, (y1 + y2) / 2.0)
             for x1, y1, x2, y2, _ in dets]
        )

        # ── Per-detection adaptive thresholds ────────
        det_heights = np.array(
            [max(y2 - y1, 1.0) for _, y1, _, y2, _ in dets]
        )
        if adaptive_tau:
            tau_per_det = np.maximum(tau_floor,
                                     tau_fraction * det_heights)
        else:
            tau_per_det = np.full(N, tau_match)

        # ── Project BEV tracks ──────
        projected, valid = project_tracks_to_image(
            bev_track_positions, H, img_w, img_h,
            margin=projection_margin,
        )

        if verbose:
            n_valid = int(valid.sum())
            print(f"    Projected: {n_valid}/{M} valid  "
                  f"(margin={projection_margin})")
            for i in range(min(M, 5)):
                print(f"      Track {bev_track_ids[i]}: "
                      f"grid=({bev_track_positions[i][0]:.1f},"
                      f"{bev_track_positions[i][1]:.1f}) → "
                      f"px=({projected[i][0]:.1f},"
                      f"{projected[i][1]:.1f}) "
                      f"valid={valid[i]}")
            for j in range(min(N, 5)):
                tau_j = tau_per_det[j]
                print(f"      Det {j}: ref=({det_ref[j][0]:.1f},"
                      f"{det_ref[j][1]:.1f}) "
                      f"h={det_heights[j]:.0f}px "
                      f"τ={tau_j:.0f}px "
                      f"score={dets[j][4]:.2f}")

        # ── Build cost matrix with gating ───────────
        cost = np.full((N, M), 1e9)
        for i in range(M):
            if valid[i]:
                dists = np.linalg.norm(
                    det_ref - projected[i:i + 1], axis=1
                )
                # Gate: forbid entries beyond generous multiple
                gate = cost_gate_factor * tau_per_det
                cost[:, i] = np.where(dists <= gate, dists, 1e9)

        # ── Hungarian matching ────────────────────────────
        row_ind, col_ind = linear_sum_assignment(cost)

        cam_results = []
        matched_dets = set()
        n_matched = 0

        for r, c in zip(row_ind, col_ind):
            x1, y1, x2, y2, s = dets[r]
            effective_tau = tau_per_det[r]
            if cost[r, c] < effective_tau:
                cam_results.append(
                    (x1, y1, x2, y2, s, bev_track_ids[c])
                )
                matched_dets.add(r)
                n_matched += 1
                if verbose:
                    print(f"    MATCH: det {r} ↔ track "
                          f"{bev_track_ids[c]} "
                          f"(dist={cost[r, c]:.1f}px, "
                          f"τ={effective_tau:.0f}px)")
            else:
                if verbose and cost[r, c] < 1e8:
                    print(f"    REJECT: det {r} ↔ track "
                          f"{bev_track_ids[c]} "
                          f"(dist={cost[r, c]:.1f}px > "
                          f"τ={effective_tau:.0f}px)")

        # Append unmatched detections with tid = -1
        for j, (x1, y1, x2, y2, s) in enumerate(dets):
            if j not in matched_dets:
                cam_results.append((x1, y1, x2, y2, s, -1))

        if verbose:
            # ── Diagnostic: offset vector summary ─────────
            match_offsets = []
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < tau_per_det[r]:
                    dx = det_ref[r, 0] - projected[c, 0]
                    dy = det_ref[r, 1] - projected[c, 1]
                    match_offsets.append((dx, dy))
            if match_offsets:
                offs = np.array(match_offsets)
                print(f"    Offset μ=({offs[:, 0].mean():.1f},"
                      f"{offs[:, 1].mean():.1f}) "
                      f"σ=({offs[:, 0].std():.1f},"
                      f"{offs[:, 1].std():.1f})")
            print(f"    → {n_matched} matched, "
                  f"{len(dets) - n_matched} unmatched")

        results[cam_idx] = cam_results

    return results


def visualize_2d_tracking(
    cam_paths, cam_results, homographies, bev_positions, bev_ids,
    save_dir, frame_id, seq_id, img_hw,
    use_foot_point=True,
):
    """
    Save annotated images showing:
    - YOLO detections (green = matched, red = unmatched)
    - Projected BEV track positions (blue circles)
    - Track IDs
    - Match lines connecting projected points to detection ref-points
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("WARNING: PIL not available for visualization")
        return

    img_h, img_w = img_hw

    for cam_idx, path in enumerate(cam_paths):
        if not os.path.exists(path):
            continue

        img = Image.open(path).convert('RGB')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                size=max(24, img.width // 100),
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

        H = homographies[cam_idx]

        # ── Draw projected BEV tracks (blue circles) ─────
        proj_uv = {}
        if bev_positions:
            projected, valid = project_tracks_to_image(
                bev_positions, H, img.width, img.height,
                margin=0.02,
            )
            for i, (uv, v) in enumerate(zip(projected, valid)):
                if v:
                    cx, cy = int(uv[0]), int(uv[1])
                    r = max(12, img.width // 200)
                    draw.ellipse(
                        [cx - r, cy - r, cx + r, cy + r],
                        outline='blue', width=3,
                    )
                    draw.text(
                        (cx + r + 4, cy - r),
                        f"BEV:{bev_ids[i]}",
                        fill='blue', font=font,
                    )
                    proj_uv[bev_ids[i]] = (cx, cy)

        # ── Draw YOLO detections ─────────────────────────
        cam_dets = cam_results.get(cam_idx, [])
        for (x1, y1, x2, y2, score, tid) in cam_dets:
            if tid >= 0:
                color = 'lime'
                label = f"ID:{tid} ({score:.2f})"
            else:
                color = 'red'
                label = f"? ({score:.2f})"

            lw = max(3, img.width // 500)
            draw.rectangle(
                [int(x1), int(y1), int(x2), int(y2)],
                outline=color, width=lw,
            )
            draw.text(
                (int(x1), int(y1) - max(28, img.width // 80)),
                label, fill=color, font=font,
            )

            # ── Draw reference point (foot-point or centre) ──
            if use_foot_point:
                ref_x = int((x1 + x2) / 2)
                ref_y = int(y2)
            else:
                ref_x = int((x1 + x2) / 2)
                ref_y = int((y1 + y2) / 2)
            dot_r = max(6, img.width // 400)
            ref_color = 'lime' if tid >= 0 else 'orange'
            draw.ellipse(
                [ref_x - dot_r, ref_y - dot_r,
                 ref_x + dot_r, ref_y + dot_r],
                fill=ref_color,
            )

            # ── Draw match line: projected → ref-point ───
            if tid >= 0 and tid in proj_uv:
                pcx, pcy = proj_uv[tid]
                draw.line(
                    [(pcx, pcy), (ref_x, ref_y)],
                    fill='cyan', width=max(2, lw // 2),
                )
                mid_x = (pcx + ref_x) // 2
                mid_y = (pcy + ref_y) // 2
                dist = ((pcx - ref_x)**2 + (pcy - ref_y)**2)**0.5
                draw.text(
                    (mid_x, mid_y),
                    f"{dist:.0f}px",
                    fill='cyan', font=font,
                )

        # ── Save ─────────────────────────────────────────
        out_dir = os.path.join(
            save_dir, f'seq{seq_id}', f'cam{cam_idx}'
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'frame_{frame_id:06d}.jpg')
        img.save(out_path, quality=85)

def compute_matching_diagnostics(
    bev_track_positions,
    bev_track_ids,
    detections_per_cam,
    homographies,
    img_hw,
    tau_score=0.3,
    projection_margin=0.02,
):
    """
    Compute per-camera matching diagnostics WITHOUT running the
    full assignment.  Returns a dict of per-camera statistics useful
    for logging and debugging.

    Returns
    -------
    dict cam_idx → {
        'n_valid_tracks'  : int,
        'n_dets'          : int,
        'foot_dists'      : ndarray (N_valid_tracks, N_dets) of foot-point
                            distances,
        'centre_dists'    : ndarray  ... of centre-point distances,
        'mean_foot_offset': (float, float)  mean (dx,dy) foot-vs-centre
                            across detections,
        'det_heights'     : ndarray of bbox heights,
    }
    """
    M = len(bev_track_positions)
    img_h, img_w = img_hw
    diagnostics = {}

    for cam_idx, H in enumerate(homographies):
        raw_dets = detections_per_cam.get(cam_idx, [])
        dets = [(x1, y1, x2, y2, s)
                for x1, y1, x2, y2, s in raw_dets if s >= tau_score]

        if not dets or M == 0:
            diagnostics[cam_idx] = {
                'n_valid_tracks': 0, 'n_dets': len(dets),
                'foot_dists': np.empty((0, 0)),
                'centre_dists': np.empty((0, 0)),
                'mean_foot_offset': (0.0, 0.0),
                'det_heights': np.array([]),
            }
            continue

        projected, valid = project_tracks_to_image(
            bev_track_positions, H, img_w, img_h,
            margin=projection_margin,
        )
        valid_idx = np.where(valid)[0]
        proj_valid = projected[valid_idx]

        det_feet = np.array([((x1+x2)/2.0, y2)
                             for x1, y1, x2, y2, _ in dets])
        det_centres = np.array([((x1+x2)/2.0, (y1+y2)/2.0)
                                for x1, y1, x2, y2, _ in dets])
        det_heights = np.array([y2-y1 for _, y1, _, y2, _ in dets])

        N = len(dets)
        Mv = len(valid_idx)

        foot_dists = np.full((Mv, N), np.inf)
        centre_dists = np.full((Mv, N), np.inf)
        for i in range(Mv):
            foot_dists[i] = np.linalg.norm(
                det_feet - proj_valid[i:i+1], axis=1)
            centre_dists[i] = np.linalg.norm(
                det_centres - proj_valid[i:i+1], axis=1)

        # Mean offset from centre to foot (should show systematic
        # vertical bias when using centres)
        foot_offset_dx = det_feet[:, 0] - det_centres[:, 0]
        foot_offset_dy = det_feet[:, 1] - det_centres[:, 1]

        diagnostics[cam_idx] = {
            'n_valid_tracks': Mv,
            'n_dets': N,
            'foot_dists': foot_dists,
            'centre_dists': centre_dists,
            'mean_foot_offset': (
                float(foot_offset_dx.mean()),
                float(foot_offset_dy.mean()),
            ),
            'det_heights': det_heights,
        }

    return diagnostics

def evaluate_2d_mot(gt_list, pred_list, iou_threshold=0.5):
    """
    Evaluate 2D multi-object tracking using *motmetrics*.

    Both lists contain rows of the form:
        gt:   [frame, id, x, y, w, h]
        pred: [frame, id, x, y, w, h, score]

    Returns a pandas Series, or None if motmetrics is unavailable.
    """
    try:
        import motmetrics as mm
    except ImportError:
        print("WARNING: motmetrics not installed – "
              "skipping 2D MOT eval. Install: pip install motmetrics")
        return None

    acc = mm.MOTAccumulator(auto_id=True)

    gt_arr = np.array(gt_list) if gt_list else np.empty((0, 6))
    pred_arr = np.array(pred_list) if pred_list else np.empty((0, 7))

    all_frames = set()
    if len(gt_arr):
        all_frames.update(gt_arr[:, 0].astype(int).tolist())
    if len(pred_arr):
        all_frames.update(pred_arr[:, 0].astype(int).tolist())

    for frame in sorted(all_frames):
        gt_f = (gt_arr[gt_arr[:, 0].astype(int) == frame]
                if len(gt_arr) else np.empty((0, 6)))
        pr_f = (pred_arr[pred_arr[:, 0].astype(int) == frame]
                if len(pred_arr) else np.empty((0, 7)))

        gt_ids = gt_f[:, 1].astype(int).tolist() if len(gt_f) else []
        pr_ids = pr_f[:, 1].astype(int).tolist() if len(pr_f) else []

        if len(gt_f) > 0 and len(pr_f) > 0:
            gt_boxes = gt_f[:, 2:6].astype(float)
            pr_boxes = pr_f[:, 2:6].astype(float)
            dists = mm.distances.iou_matrix(
                gt_boxes, pr_boxes, max_iou=1 - iou_threshold
            )
        elif len(gt_f) > 0:
            dists = np.full((len(gt_f), 0), np.nan)
        elif len(pr_f) > 0:
            dists = np.full((0, len(pr_f)), np.nan)
        else:
            continue

        acc.update(gt_ids, pr_ids, dists)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            'mota', 'motp', 'idf1',
            'num_switches', 'num_false_positives', 'num_misses',
            'mostly_tracked', 'mostly_lost', 'num_objects',
        ],
        name='OVERALL',
    )
    return summary.iloc[0] if len(summary) else None