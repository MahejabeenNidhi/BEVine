#!/usr/bin/env python
# WorldTrack/evaluation/obb_iou.py
"""
Oriented-BEV-box (OBB) IoU utilities + standalone evaluator.

Box convention:  (cx, cy, length, width, yaw)
  * cx, cy, length, width in the SAME unit (cm, as written by world_track.py)
  * length is along the local +x axis, width along local +y
  * yaw is a rotation about the world up-axis, radians

File formats (produced by WorldTrackModel.on_test_epoch_end):
  gt_OBB_moda.txt / pred_OBB_moda.txt   : frame x y yaw length width
  gt_OBB_mota.txt / pred_OBB_mota.txt   : seq,frame,id,x,y,yaw,length,width
"""
import argparse
import numpy as np


# ──────────────────────────────────────────────────────────────
# geometry
# ──────────────────────────────────────────────────────────────
def obb_corners(cx, cy, length, width, yaw):
    """Return the 4 corners (4,2) of an oriented rectangle."""
    dx, dy = length / 2.0, width / 2.0
    pts = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]],
                   dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pts @ R.T + np.array([cx, cy], dtype=np.float64)


def _signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ccw(poly):
    return poly if _signed_area(poly) >= 0 else poly[::-1].copy()


def _clip_polygon(subject, clipper):
    """Sutherland-Hodgman: clip convex `subject` by convex `clipper`."""
    out = _ccw(np.asarray(subject, dtype=np.float64))
    clip = _ccw(np.asarray(clipper, dtype=np.float64))
    for i in range(len(clip)):
        if len(out) == 0:
            return np.zeros((0, 2))
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a
        inp, out = out, []
        for j in range(len(inp)):
            cur, nxt = inp[j], inp[(j + 1) % len(inp)]
            cs = edge[0] * (cur[1] - a[1]) - edge[1] * (cur[0] - a[0])
            ns = edge[0] * (nxt[1] - a[1]) - edge[1] * (nxt[0] - a[0])
            if cs >= 0:
                out.append(cur)
            if (cs >= 0) != (ns >= 0):
                den = cs - ns
                if abs(den) > 1e-12:
                    out.append(cur + (cs / den) * (nxt - cur))
        out = (np.array(out, dtype=np.float64) if out
               else np.zeros((0, 2)))
    return out


def obb_iou(box_a, box_b):
    """IoU of two oriented boxes (cx, cy, length, width, yaw)."""
    pa = obb_corners(*box_a)
    pb = obb_corners(*box_b)
    inter_poly = _clip_polygon(pa, pb)
    if len(inter_poly) < 3:
        return 0.0
    inter = abs(_signed_area(inter_poly))
    area_a, area_b = abs(_signed_area(pa)), abs(_signed_area(pb))
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def circumradius(length, width):
    """Radius of the circle that circumscribes the rectangle."""
    return 0.5 * np.hypot(length, width)


def obb_iou_matrix(pred, gt, prefilter=True):
    """pred (P,5), gt (G,5) -> (P,G) IoU matrix.

    PREFILTER (equivalence proof)
    -----------------------------
    Every rectangle is fully contained in the circle centred on its own
    centre with radius r = 0.5*sqrt(length^2 + width^2). If the centre
    distance d satisfies d >= r_pred + r_gt the two circles are disjoint
    (or touch in a single point), hence the two rectangles are disjoint,
    hence their intersection area is exactly 0 and IoU is exactly 0.0 --
    which is precisely the value the matrix is initialised with.
    Therefore the prefilter can never discard a pair with IoU > 0, and a
    fortiori never a pair whose IoU exceeds any positive threshold.
    Set prefilter=False to force the exhaustive O(P*G) polygon clipping
    (used by tests/test_obb_metrics.py to assert bit-equality).
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 5)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 5)
    P, G = len(pred), len(gt)
    M = np.zeros((P, G), dtype=np.float64)
    if P == 0 or G == 0:
        return M

    if prefilter:
        rp = circumradius(pred[:, 2], pred[:, 3])
        rg = circumradius(gt[:, 2], gt[:, 3])
        d = np.hypot(pred[:, None, 0] - gt[None, :, 0],
                     pred[:, None, 1] - gt[None, :, 1])
        cand = d < (rp[:, None] + rg[None, :])
    else:
        cand = np.ones((P, G), dtype=bool)

    for i, j in np.argwhere(cand):
        M[i, j] = obb_iou(pred[i], gt[j])
    return M


# ──────────────────────────────────────────────────────────────
# evaluation
# ──────────────────────────────────────────────────────────────
def evaluate(gt_path, pred_path, iou_thresh=0.3, delimiter=None,
             frame_col=0, box_cols=(1, 2, 4, 5, 3)):
    """Standalone OBB CLEAR-MOD evaluation.

    NOTE 1: this now uses the SAME optimal (Hungarian) association as the
            in-run evaluation in world_track.py, so CLI numbers and
            training-run numbers agree. The old greedy matcher remains
            available as evaluation.obb_metrics.greedy_match (tests only).
    NOTE 2: box_cols must index (x, y, length, width, yaw). The previous
            default (1,2,3,4,5) silently fed 'yaw' in as 'length' for the
            *_OBB_moda.txt layout (frame x y yaw length width); the
            default is now the correct (1,2,4,5,3).
    """
    try:
        from evaluation.obb_metrics import (
            rows_to_frame_dict, obb_mod_metrics, print_mod_report)
    except ImportError:
        from obb_metrics import (
            rows_to_frame_dict, obb_mod_metrics, print_mod_report)

    gt = np.loadtxt(gt_path, delimiter=delimiter, ndmin=2)
    try:
        pred = np.loadtxt(pred_path, delimiter=delimiter, ndmin=2)
    except (OSError, ValueError):
        pred = np.empty((0, gt.shape[1]))

    res = obb_mod_metrics(
        rows_to_frame_dict(gt, frame_col, box_cols),
        rows_to_frame_dict(pred, frame_col, box_cols),
        iou_thresh,
    )
    print_mod_report(res)
    return res


def selftest():
    a = (0.0, 0.0, 100.0, 100.0, 0.0)
    assert abs(obb_iou(a, a) - 1.0) < 1e-9
    # square rotated 90° == itself
    assert abs(obb_iou(a, (0., 0., 100., 100., np.pi / 2)) - 1.0) < 1e-6
    # 50 % shift along x -> 5000 / 15000
    assert abs(obb_iou(a, (50., 0., 100., 100., 0.)) - 1 / 3) < 1e-6
    # disjoint
    assert obb_iou(a, (1000., 0., 100., 100., 0.)) == 0.0
    # cow-sized box, 20° yaw error
    iou = obb_iou((0., 0., 200., 70., 0.),
                  (0., 0., 200., 70., np.radians(20)))
    print(f"selftest OK  (200x70 box, 20 deg yaw error -> IoU={iou:.3f})")


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=None)
    ap.add_argument('--pred', default=None)
    ap.add_argument('--iou', type=float, nargs='+', default=[0.25, 0.50],
                    help='one or more IoU thresholds')
    ap.add_argument('--mota', action='store_true',
                    help='files are the comma-separated *_OBB_mota.txt '
                         'format (seq,frame,id,x,y,yaw,l,w)')
    ap.add_argument('--track', action='store_true',
                    help='with --mota: also run OBB CLEAR-MOT (motmetrics)')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        selftest()

    if args.gt and args.pred:
        from evaluation.obb_metrics import (
            load_mota_files, obb_mot_metrics, print_mot_report)
        for thr in args.iou:
            if args.mota:
                # detection-style scoring of the per-track file
                evaluate(args.gt, args.pred, thr, delimiter=',',
                         frame_col=1, box_cols=(3, 4, 6, 7, 5))
            else:
                evaluate(args.gt, args.pred, thr)
        if args.mota and args.track:
            g, p = load_mota_files(args.gt, args.pred)
            for thr in args.iou:
                s = obb_mot_metrics(g, p, thr)
                if s is None:
                    print('motmetrics unavailable or no GT -> '
                          'OBB CLEAR-MOT skipped')
                    break
                print_mot_report(s, thr)
