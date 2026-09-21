#!/usr/bin/env python
# WorldTrack/evaluation/check_track_parity.py
"""P5 parity check: point track export vs OBB track export.

For every (seq, frame) the set of track IDs in mota_pred.txt must equal
the set in pred_OBB_mota.txt, the positions must agree once units are
accounted for (point export = world-grid CELLS, OBB export = world CM),
and every OBB row must carry a well-defined box (finite, L/W > 0) —
this extends the in-run `obb_tracks_missing_box` counter to a hard,
post-hoc gate.

Exit code 0 = parity holds, 1 = mismatch (fails the run loudly in CI).

Usage:
    python evaluation/check_track_parity.py \
        --log_dir <run>/lightning_logs/version_0 --cell-cm 10.0
"""
import argparse
import os
import sys

import numpy as np


def load(path, cols):
    if not os.path.exists(path):
        print(f"PARITY FAIL: missing file {path}")
        sys.exit(1)
    arr = np.loadtxt(path, delimiter=',', ndmin=2)
    if arr.size == 0:
        return np.empty((0, cols))
    if arr.shape[1] < cols:
        print(f"PARITY FAIL: {path} has {arr.shape[1]} cols, "
              f"expected >= {cols}")
        sys.exit(1)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir', required=True,
                    help='directory containing mota_pred.txt and '
                         'pred_OBB_mota.txt')
    ap.add_argument('--cell-cm', type=float, default=10.0,
                    help='cm per world-grid cell (mmCows: 10.0; check the '
                         '[EVAL-CHECK] grid_cell_cm line in the test log)')
    ap.add_argument('--pos-tol-cm', type=float, default=1.0,
                    help='position agreement tolerance in cm')
    ap.add_argument('--expected-tracks', type=int, default=0,
                    help='optional known number of persistent objects; '
                         '0 disables the check')
    ap.add_argument('--min-row-ratio', type=float, default=0.0,
                    help='optional minimum pred_rows / gt_rows ratio; '
                         '0 disables the check')
    args = ap.parse_args()

    # mota_pred.txt: seq,frame,id,-1,-1,-1,-1,score,x_cell,y_cell,-1
    pt = load(os.path.join(args.log_dir, 'mota_pred.txt'), 11)
    # pred_OBB_mota.txt: seq,frame,id,x_cm,y_cm,yaw,L,W
    ob = load(os.path.join(args.log_dir, 'pred_OBB_mota.txt'), 8)

    gt = load(os.path.join(args.log_dir, 'mota_gt.txt'), 11)

    gt_ids = set(gt[:, 2].astype(int).tolist())
    pred_ids = set(pt[:, 2].astype(int).tolist())
    row_ratio = float(len(pt)) / float(max(len(gt), 1))

    pred_gaps = []
    pred_frames = sorted(set(pt[:, 1].astype(int).tolist()))
    for tid in pred_ids:
        fs = sorted(pt[pt[:, 2] == tid, 1].astype(int).tolist())
        if len(fs) > 1:
            pred_gaps.extend((np.diff(fs) - 1).tolist())

    print(f"[COVERAGE] GT rows={len(gt)} pred rows={len(pt)} "
          f"row_ratio={row_ratio:.3f}")
    print(f"[COVERAGE] GT IDs={len(gt_ids)} pred IDs={len(pred_ids)}")
    print(f"[COVERAGE] predicted frames={len(pred_frames)} "
          f"interior prediction gaps: total={len(pred_gaps)} "
          f"max={max(pred_gaps) if pred_gaps else 0}")

    if args.expected_tracks > 0 and len(pred_ids) != args.expected_tracks:
        print(f"PARITY FAIL: expected {args.expected_tracks} persistent "
              f"track IDs, got {len(pred_ids)}")
        sys.exit(1)

    if row_ratio < args.min_row_ratio:
        print(f"PARITY FAIL: pred/GT row ratio {row_ratio:.3f} is below "
              f"the requested {args.min_row_ratio:.3f}")
        sys.exit(1)

    # ── OBB well-definedness (extends obb_tracks_missing_box) ──
    if len(ob):
        bad = (~np.isfinite(ob).all(axis=1)) | (ob[:, 6] <= 0) \
            | (ob[:, 7] <= 0)
        if bad.any():
            print(f"PARITY FAIL: {int(bad.sum())} OBB rows with missing/"
                  f"degenerate boxes, e.g. row {ob[bad][0].tolist()}")
            sys.exit(1)

    key_pt = set(zip(pt[:, 0].astype(int), pt[:, 1].astype(int),
                     pt[:, 2].astype(int)))
    key_ob = set(zip(ob[:, 0].astype(int), ob[:, 1].astype(int),
                     ob[:, 2].astype(int)))

    only_pt = key_pt - key_ob
    only_ob = key_ob - key_pt
    if only_pt or only_ob:
        print(f"PARITY FAIL: {len(only_pt)} (seq,frame,id) rows only in "
              f"point export, {len(only_ob)} only in OBB export.")
        print(f"  e.g. point-only: "
              f"{[tuple(int(v) for v in k) for k in sorted(only_pt)[:5]]}")
        print(f"  e.g. obb-only  : "
              f"{[tuple(int(v) for v in k) for k in sorted(only_ob)[:5]]}")
        sys.exit(1)

    # ── per-frame ID sets + unit-adjusted positions ──
    pt_idx = {(int(s), int(f), int(i)): (x, y)
              for s, f, i, x, y in
              zip(pt[:, 0], pt[:, 1], pt[:, 2], pt[:, 8], pt[:, 9])}
    n_frames = len(set((s, f) for s, f, _ in key_pt))
    worst = 0.0
    for s, f, i, x, y in zip(ob[:, 0], ob[:, 1], ob[:, 2],
                             ob[:, 3], ob[:, 4]):
        px, py = pt_idx[(int(s), int(f), int(i))]
        d = np.hypot(px * args.cell_cm - x, py * args.cell_cm - y)
        worst = max(worst, d)
        if d > args.pos_tol_cm:
            print(f"PARITY FAIL: seq {int(s)} frame {int(f)} id {int(i)}: "
                  f"point ({px:.2f},{py:.2f}) cells vs OBB "
                  f"({x:.1f},{y:.1f}) cm -> {d:.1f} cm apart "
                  f"(tol {args.pos_tol_cm} cm)")
            sys.exit(1)

    print(f"PARITY OK: {len(key_pt)} track-rows over {n_frames} frames; "
          f"ID sets identical, worst position disagreement "
          f"{worst:.2f} cm, all OBB boxes well-defined.")
    sys.exit(0)


if __name__ == '__main__':
    main()
