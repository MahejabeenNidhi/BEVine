# WorldTrack/tracking/matching_extended.py

import numpy as np
import scipy
import lap
import torch
from scipy.spatial.distance import cdist


def merge_matches(m1, m2, shape):
    """Same as original"""
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix(
        (np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix(
        (np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def linear_assignment(cost_matrix, thresh):
    """Same as original"""
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def bev_iou(boxes_a, boxes_b, box_size=30):
    """
    Compute IoU between circular regions in BEV.

    For cattle, we approximate each detection as a circular region.

    Parameters
    ----------
    boxes_a : ndarray
        Nx2 array of positions [x, y]
    boxes_b : ndarray
        Mx2 array of positions [x, y]
    box_size : float
        Radius of circular region (in cm)

    Returns
    -------
    iou : ndarray
        NxM IoU matrix
    """
    N = len(boxes_a)
    M = len(boxes_b)

    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    ious = np.zeros((N, M), dtype=np.float32)

    # Compute pairwise distances
    dists = cdist(boxes_a, boxes_b, 'euclidean')

    # Circle IoU formula
    # For two circles with radius r at distance d:
    # If d >= 2r: IoU = 0
    # If d = 0: IoU = 1
    # Otherwise: IoU = (2 * r^2 * arccos(d/2r) - d/2 * sqrt(4r^2 - d^2)) / (pi * r^2)

    r = box_size
    mask_overlap = dists < 2 * r

    for i in range(N):
        for j in range(M):
            if not mask_overlap[i, j]:
                ious[i, j] = 0.0
            elif dists[i, j] < 1e-6:  # Same position
                ious[i, j] = 1.0
            else:
                d = dists[i, j]
                # Intersection area
                intersection = (2 * r ** 2 * np.arccos(d / (2 * r)) -
                                d / 2 * np.sqrt(4 * r ** 2 - d ** 2))
                # Union area (two circles)
                union = 2 * np.pi * r ** 2 - intersection
                ious[i, j] = intersection / union

    return ious


def iou_distance(atracks, btracks, box_size=30):
    """
    Compute cost based on BEV IoU

    Parameters
    ----------
    atracks : list
        List of positions [x, y] or STrack objects
    btracks : list
        List of positions [x, y] or STrack objects
    box_size : float
        Radius of cattle bounding region

    Returns
    -------
    cost_matrix : ndarray
        Cost matrix (1 - IoU)
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)):
        atlocs = atracks
    else:
        atlocs = np.array([track.xy for track in atracks])

    if (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        btlocs = btracks
    else:
        btlocs = np.array([track.xy for track in btracks])

    _ious = bev_iou(atlocs, btlocs, box_size)
    cost_matrix = 1 - _ious

    return cost_matrix


def center_distance(atracks, btracks):
    """Same as original"""
    cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=np.float32)

    if cost_matrix.size == 0:
        return cost_matrix

    atracks = np.stack(atracks)
    btracks = np.stack(btracks)
    cost_matrix = cdist(atracks, btracks, 'euclidean')

    return cost_matrix


def trajectory_distance(atracks, btracks, buffer_size=10):
    """
    Compute distance based on trajectory history (buffered matching).

    Parameters
    ----------
    atracks : list[STrack]
        List of tracks with history
    btracks : list[STrack]
        List of detections
    buffer_size : int
        Number of historical positions to consider

    Returns
    -------
    cost_matrix : ndarray
        Distance based on trajectory similarity
    """
    cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=np.float32)

    if cost_matrix.size == 0:
        return cost_matrix

    for i, track in enumerate(atracks):
        # Get track's trajectory history - convert deque to list
        position_history = getattr(track, 'position_history', None)
        if position_history is not None:
            track_history = list(position_history)[-buffer_size:]
        else:
            track_history = [track.xy]

        # Get track's predicted trajectory
        track_predictions = getattr(track, 'predicted_positions', [])

        for j, det in enumerate(btracks):
            # Handle both numpy arrays and tensors for detection position
            if isinstance(det, np.ndarray):
                det_pos = det
            elif hasattr(det, 'xy'):
                det_xy = det.xy
                # Convert tensor to numpy if needed
                if isinstance(det_xy, torch.Tensor):
                    det_pos = det_xy.cpu().numpy() if det_xy.is_cuda else det_xy.numpy()
                else:
                    det_pos = np.array(det_xy) if not isinstance(det_xy, np.ndarray) else det_xy
            else:
                det_pos = np.array(det)

            # Ensure det_pos is 1D numpy array
            det_pos = np.asarray(det_pos).flatten()

            # Method 1: Distance to recent history
            if len(track_history) > 0:
                # Convert all history positions to numpy arrays
                hist_positions = []
                for pos in track_history:
                    if isinstance(pos, torch.Tensor):
                        pos_np = pos.cpu().numpy() if pos.is_cuda else pos.numpy()
                    else:
                        pos_np = np.array(pos) if not isinstance(pos, np.ndarray) else pos
                    hist_positions.append(pos_np.flatten())

                hist_positions = np.array(hist_positions)
                hist_distances = np.linalg.norm(hist_positions - det_pos, axis=1)
                min_hist_dist = np.min(hist_distances)
            else:
                min_hist_dist = np.inf

            # Method 2: Distance to predicted position
            if len(track_predictions) > 0:
                pred_pos = track_predictions[0]  # Next predicted position
                # Convert tensor to numpy if needed
                if isinstance(pred_pos, torch.Tensor):
                    pred_pos = pred_pos.cpu().numpy() if pred_pos.is_cuda else pred_pos.numpy()
                pred_pos = np.asarray(pred_pos).flatten()
                pred_dist = np.linalg.norm(pred_pos - det_pos)
            else:
                pred_dist = np.inf

            # Combine both measures
            # Weight prediction more than history
            cost_matrix[i, j] = 0.7 * pred_dist + 0.3 * min_hist_dist

    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections,
                only_position=True, lambda_=0.98, gating_threshold=1000):
    """Enhanced version with adaptive gating"""
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4

    measurements = np.asarray([det.to_xyah() if hasattr(det, 'to_xyah')
                               else np.r_[det, 1, 1] for det in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements,
            only_position, metric='maha')

        # Adaptive gating based on track age
        track_age = getattr(track, 'tracklet_len', 1)
        adaptive_threshold = gating_threshold * (1 + 0.1 * min(track_age, 10))

        cost_matrix[row, gating_distance > adaptive_threshold] = np.inf
        cost_matrix[row] = (lambda_ * cost_matrix[row] +
                            (1 - lambda_) * gating_distance * 0.1)

    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections,
                     only_position=False):
    """Same as original"""
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    from tracking.kalman_filter import chi2inv95
    gating_threshold = chi2inv95[gating_dim]

    measurements = np.asarray([det.to_xyah() if hasattr(det, 'to_xyah')
                               else np.r_[det, 1, 1] for det in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf

    return cost_matrix