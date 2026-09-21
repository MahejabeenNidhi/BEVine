#!/usr/bin/env python
# WorldTrack/evaluation/obb_metrics.py
"""
Oriented-bounding-box (OBB) CLEAR-MOD / CLEAR-MOT evaluation.

=====================  CONVENTIONS (read me)  =========================
Box            : (cx, cy, length, width, yaw)
                 length is along the box's LOCAL +x axis,
                 width  is along the box's LOCAL +y axis.
Units          : world CENTIMETRES, BEV (ground) plane. Identical to
                 the units written by WorldTrackModel into
                 gt_OBB_moda.txt / pred_OBB_moda.txt and
                 gt_OBB_mota.txt / pred_OBB_mota.txt.
Angle          : yaw in RADIANS, CCW positive, in the BEV plane.
Similarity     : rotated BEV IoU (evaluation.obb_iou.obb_iou).
Association    : OPTIMAL (Hungarian) assignment on cost = 1 - IoU, with
                 pairs whose IoU < threshold FORBIDDEN. This mirrors how
                 CLEAR_MOD_HUN / motmetrics treat the Euclidean gate in
                 the point-based path, but with an overlap gate.
MODA           : 1 - (FN + FP) / N_gt          (reported in %, NOT
                 clamped at 0 -- unlike evaluation/CLEAR_MOD_HUN.py).
MODP           : mean IoU over matched (TP) pairs, reported in %.
                 (Overlap-based MODP: the standard variant whenever the
                 association itself is overlap-based.)
MOTP           : motmetrics returns mean distance = mean(1 - IoU);
                 we report 100 * (1 - motp) = mean IoU over matches, %.

!! IoU of a rotated rectangle is INVARIANT to a 180-degree yaw flip.  !!
!! Heading correctness therefore remains the responsibility of the    !!
!! existing pose3d/yaw_* metrics, not of these numbers.               !!

!! The point-based MODA/MOTA use a EUCLIDEAN centre-distance gate;    !!
!! these OBB numbers use an IoU gate. They are DIFFERENT association  !!
!! criteria and the two families of numbers are NOT directly          !!
!! comparable.                                                        !!
=======================================================================
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

try:                                    # normal package import
    from evaluation.obb_iou import obb_iou, obb_iou_matrix, obb_corners
except ImportError:                     # running from inside evaluation/
    from obb_iou import obb_iou, obb_iou_matrix, obb_corners  # noqa: F401

DEFAULT_IOU_THRESHOLDS = (0.25, 0.50)

# Same trick as evaluation/CLEAR_MOD_HUN.py: a huge finite cost instead
# of np.inf, because scipy's Hungarian is extremely slow with inf.
_FORBIDDEN_COST = 1e6
_EMPTY = np.zeros((0, 5), dtype=np.float64)


# ──────────────────────────────────────────────────────────────
# small helpers
# ──────────────────────────────────────────────────────────────
def box_areas(boxes):
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 5)
    return np.abs(b[:, 2] * b[:, 3])


def rows_to_frame_dict(arr, frame_col, box_cols):
    """(N, C) rows -> {int frame: (n, 5) boxes}.

    box_cols must index (x, y, length, width, yaw) IN THAT ORDER.
    """
    out = {}
    if arr is None or len(arr) == 0:
        return out
    arr = np.asarray(arr, dtype=np.float64)
    fr = arr[:, frame_col]
    for f in np.unique(fr):
        out[int(round(float(f)))] = arr[fr == f][:, list(box_cols)]
    return out


def rows_to_seq_frame_dict(arr, seq_col, frame_col, id_col, box_cols):
    """(N, C) rows -> {int seq: {int frame: (ids (n,), boxes (n,5))}}."""
    out = {}
    if arr is None or len(arr) == 0:
        return out
    arr = np.asarray(arr, dtype=np.float64)
    for s in np.unique(arr[:, seq_col]):
        rows_s = arr[arr[:, seq_col] == s]
        per_frame = {}
        for f in np.unique(rows_s[:, frame_col]):
            r = rows_s[rows_s[:, frame_col] == f]
            per_frame[int(round(float(f)))] = (
                r[:, id_col].astype(np.int64),
                r[:, list(box_cols)],
            )
        out[int(round(float(s)))] = per_frame
    return out


# ──────────────────────────────────────────────────────────────
# association
# ──────────────────────────────────────────────────────────────
def hungarian_match(iou_mat, iou_thresh):
    """Optimal assignment on cost = 1 - IoU, sub-threshold forbidden.

    Returns [(pred_idx, gt_idx), ...] with IoU >= iou_thresh.
    """
    iou_mat = np.asarray(iou_mat, dtype=np.float64)
    if iou_mat.size == 0:
        return []
    cost = 1.0 - iou_mat                       # new array, input untouched
    cost[iou_mat < iou_thresh] = _FORBIDDEN_COST
    ri, ci = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(ri, ci)
            if iou_mat[i, j] >= iou_thresh]


def greedy_match(iou_mat, iou_thresh):
    """Highest-IoU-first matching. Kept ONLY as a reference for the
    unit tests; no reported metric uses it."""
    iou_mat = np.asarray(iou_mat, dtype=np.float64)
    if iou_mat.size == 0:
        return []
    used_p, used_g, out = set(), set(), []
    for idx in np.argsort(-iou_mat, axis=None):
        i, j = np.unravel_index(idx, iou_mat.shape)
        if iou_mat[i, j] < iou_thresh:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(int(i)); used_g.add(int(j))
        out.append((int(i), int(j)))
    return out

def nms_obb(boxes, scores, iou_thresh=0.5):
    """Greedy rotated-IoU NMS. Returns kept indices (score-descending).

    The heat-map max_pool2d NMS in utils/decode.py only suppresses a peak
    if a higher peak sits within ONE BEV cell. For a 180 cm animal on a
    10 cm grid, two peaks 3 cells apart survive that filter yet overlap by
    ~80% as boxes. Every survivor is a guaranteed FP under one-to-one
    matching, so box-level NMS is required for a fair OBB score.
    """
    boxes = np.asarray(boxes, np.float64).reshape(-1, 5)
    scores = np.asarray(scores, np.float64).reshape(-1)
    if len(boxes) == 0:
        return []
    r = 0.5 * np.hypot(boxes[:, 2], boxes[:, 3])
    keep = []
    for i in np.argsort(-scores):
        i = int(i)
        drop = False
        for j in keep:
            d = np.hypot(boxes[i, 0] - boxes[j, 0], boxes[i, 1] - boxes[j, 1])
            if d >= r[i] + r[j]:          # provably IoU == 0, skip clipping
                continue
            if obb_iou(boxes[i], boxes[j]) >= iou_thresh:
                drop = True
                break
        if not drop:
            keep.append(i)
    return keep

# ──────────────────────────────────────────────────────────────
# CLEAR-MOD (detection)
# ──────────────────────────────────────────────────────────────
def obb_mod_metrics(gt_by_frame, pred_by_frame, iou_thresh):
    """Overlap-gated CLEAR-MOD. Frames present in only one of the two
    dicts are handled (all-FN / all-FP)."""
    frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    tp = fp = fn = 0
    matched_ious, per_frame = [], []
    n_pred_total, n_degenerate = 0, 0

    for f in frames:
        g = np.asarray(gt_by_frame.get(f, _EMPTY), dtype=np.float64).reshape(-1, 5)
        p = np.asarray(pred_by_frame.get(f, _EMPTY), dtype=np.float64).reshape(-1, 5)
        n_pred_total += len(p)
        n_degenerate += int(np.sum(box_areas(g) <= 0.0)) \
            + int(np.sum(box_areas(p) <= 0.0))

        if len(g) == 0:
            fp += len(p)
            per_frame.append((f, 0, len(p), 0))
            continue
        if len(p) == 0:
            fn += len(g)
            per_frame.append((f, 0, 0, len(g)))
            continue

        M = obb_iou_matrix(p, g)                       # (P, G)
        matches = hungarian_match(M, iou_thresh)
        tp_f = len(matches)
        matched_ious.extend(float(M[i, j]) for i, j in matches)
        tp += tp_f
        fp += len(p) - tp_f
        fn += len(g) - tp_f
        per_frame.append((f, tp_f, len(p) - tp_f, len(g) - tp_f))

    n_gt = tp + fn
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return {
        'iou_thresh': float(iou_thresh),
        'n_frames': len(frames),
        'n_gt': int(n_gt),
        'n_pred': int(n_pred_total),
        'n_degenerate_boxes': int(n_degenerate),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'recall': 100.0 * tp / n_gt if n_gt else 0.0,
        'precision': 100.0 * tp / (tp + fp) if (tp + fp) else 0.0,
        'moda': 100.0 * (1.0 - (fp + fn) / n_gt) if n_gt else 0.0,
        'modp': 100.0 * mean_iou,          # mean IoU over TP, in %
        'mean_iou': mean_iou,
        'min_iou': float(np.min(matched_ious)) if matched_ious else 0.0,
        'per_frame': per_frame,
    }


# ──────────────────────────────────────────────────────────────
# CLEAR-MOT (tracking) -- delegated to motmetrics
# ──────────────────────────────────────────────────────────────
def obb_mot_metrics(gt_by_seq, pred_by_seq, iou_thresh):
    """One motmetrics accumulator per sequence (IDs never mix across
    sequences). Distance fed to motmetrics is 1 - IoU, with sub-threshold
    pairs set to np.nan (= 'not allowed'), exactly how mot_bev.py uses
    the max_d2-clipped Euclidean distance today.

    Returns the motmetrics summary DataFrame (index = seq names +
    'OVERALL'), or None if motmetrics is unavailable / there is no GT.
    """
    try:
        import motmetrics as mm
    except ImportError:
        return None

    accs, names = [], []
    for seq in sorted(set(gt_by_seq) | set(pred_by_seq)):
        g_frames = gt_by_seq.get(seq, {})
        p_frames = pred_by_seq.get(seq, {})
        if not g_frames:                      # no 3D GT for this sequence
            continue
        acc = mm.MOTAccumulator(auto_id=False)
        for f in sorted(set(g_frames) | set(p_frames)):
            gids, gboxes = g_frames.get(f, (np.zeros(0, np.int64), _EMPTY))
            pids, pboxes = p_frames.get(f, (np.zeros(0, np.int64), _EMPTY))
            if len(gids) and len(pids):
                M = obb_iou_matrix(pboxes, gboxes)     # (P, G)
                C = 1.0 - M.T                          # (G, P) -> motmetrics
                C[M.T < iou_thresh] = np.nan
            else:
                C = np.full((len(gids), len(pids)), np.nan, dtype=np.float64)
            acc.update([int(i) for i in gids],
                       [int(i) for i in pids],
                       C, frameid=int(f))
        accs.append(acc)
        names.append(f'seq{int(seq)}')

    if not accs:
        return None
    mh = mm.metrics.create()
    return mh.compute_many(accs, names=names,
                           metrics=mm.metrics.motchallenge_metrics,
                           generate_overall=True)


# ──────────────────────────────────────────────────────────────
# file front-ends (used by the CLI in obb_iou.py)
# ──────────────────────────────────────────────────────────────
def _loadtxt(path, delimiter=None, ncols=1):
    try:
        a = np.loadtxt(path, delimiter=delimiter, ndmin=2)
    except (OSError, ValueError):
        return np.zeros((0, ncols))
    return a if a.size else np.zeros((0, ncols))


def load_moda_files(gt_path, pred_path):
    """gt/pred_OBB_moda.txt : frame x y yaw length width (whitespace)."""
    gt = _loadtxt(gt_path, None, 6)
    pr = _loadtxt(pred_path, None, 6)
    cols = (1, 2, 4, 5, 3)                   # -> x, y, length, width, yaw
    return (rows_to_frame_dict(gt, 0, cols),
            rows_to_frame_dict(pr, 0, cols))


def load_mota_files(gt_path, pred_path):
    """gt/pred_OBB_mota.txt : seq,frame,id,x,y,yaw,length,width (comma)."""
    gt = _loadtxt(gt_path, ',', 8)
    pr = _loadtxt(pred_path, ',', 8)
    cols = (3, 4, 6, 7, 5)                   # -> x, y, length, width, yaw
    return (rows_to_seq_frame_dict(gt, 0, 1, 2, cols),
            rows_to_seq_frame_dict(pr, 0, 1, 2, cols))


def print_mod_report(res, title='OBB DETECTION (CLEAR-MOD)'):
    t = res['iou_thresh']
    print("-" * 66)
    print(f"{title}  |  IoU threshold = {t:.2f}  |  Hungarian assignment")
    print("-" * 66)
    print(f"  frames                : {res['n_frames']}")
    print(f"  GT boxes / pred boxes : {res['n_gt']} / {res['n_pred']}")
    print(f"  TP / FP / FN          : {res['tp']} / {res['fp']} / {res['fn']}")
    print(f"  Recall                : {res['recall']:.2f}%")
    print(f"  Precision             : {res['precision']:.2f}%")
    print(f"  OBB-MODA              : {res['moda']:.2f}%")
    print(f"  OBB-MODP (mean IoU)   : {res['modp']:.2f}%")
    print(f"  min IoU over TP       : {res['min_iou']:.3f}")
    if res['n_degenerate_boxes']:
        print(f"  degenerate (zero-area) boxes seen: "
              f"{res['n_degenerate_boxes']}")
    print("-" * 66)


def print_mot_report(summary, iou_thresh):
    print("-" * 66)
    print(f"OBB TRACKING (CLEAR-MOT)  |  IoU threshold = {iou_thresh:.2f}")
    print("-" * 66)
    print(f"{'name':<12}{'MOTA%':>9}{'MOTP%':>9}{'IDF1%':>9}"
          f"{'IDs':>7}{'MT':>6}{'ML':>6}{'FM':>6}")
    for name, row in summary.iterrows():
        motp = row['motp']
        motp_iou = 100.0 * (1.0 - motp) if motp == motp else float('nan')
        print(f"{str(name):<12}{100.0 * row['mota']:>9.2f}{motp_iou:>9.2f}"
              f"{100.0 * row['idf1']:>9.2f}{int(row['num_switches']):>7}"
              f"{int(row['mostly_tracked']):>6}{int(row['mostly_lost']):>6}"
              f"{int(row['num_fragmentations']):>6}")
    print("  MOTP is reported as MEAN IoU over matched pairs (higher = better).")
    print("-" * 66)
