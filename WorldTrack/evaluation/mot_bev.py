import os
import motmetrics as mm
import numpy as np

def _load_mot_file(path, required, label):
    """Load a MOT text file as a guaranteed 2-D array.

    np.loadtxt returns a 1-D array for an empty file and also for a file
    containing exactly one row.  The metric code below expects 2-D input.
    """
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"Required {label} MOT file does not exist: {path}"
            )
        return np.empty((0, 10), dtype=float)

    if os.path.getsize(path) == 0:
        if required:
            raise ValueError(
                f"Required {label} MOT file is empty: {path}"
            )
        return np.empty((0, 10), dtype=float)

    arr = np.loadtxt(path, delimiter=',', ndmin=2)

    if arr.size == 0:
        if required:
            raise ValueError(
                f"Required {label} MOT file contains no rows: {path}"
            )
        return np.empty((0, 10), dtype=float)

    # mot_metrics uses columns 0, 1, 2, 8 and 9.
    if arr.shape[1] < 10:
        raise ValueError(
            f"{label.capitalize()} MOT file has {arr.shape[1]} column(s), "
            f"but at least 10 are required: {path}"
        )

    return arr

def mot_metrics(tSource, gtSource, scale=0.1, max_dist_m=1.0):
    """Compute CLEAR-MOT metrics from MOT files whose x/y are grid cells.

    Args:
        scale: metres per grid cell (0.1 for mmCows' 10 cm grid).
        max_dist_m: association threshold in metres. The default 1.0 m
            is 100 cm, i.e. exactly 10 cells on a 10 cm grid.
    """
    scale = float(scale)
    max_dist_m = float(max_dist_m)
    if scale <= 0:
        raise ValueError(f"scale must be positive metres/cell, got {scale}")
    if max_dist_m <= 0:
        raise ValueError(
            f"max_dist_m must be positive metres, got {max_dist_m}"
        )

    gt = _load_mot_file(
        gtSource, required=True, label='ground-truth'
    )
    dt = _load_mot_file(
        tSource, required=False, label='prediction'
    )

    # Diagnostic for unit regressions: both numbers should describe the
    # same physical association gate (100 cm = 10 cells for mmCows).
    print(f"[MOT-CHECK] scale={scale:.3f} m/cell  "
          f"max_dist={max_dist_m:.3f} m "
          f"({max_dist_m * 100.0:.1f} cm = "
          f"{max_dist_m / scale:.2f} cells)")

    if len(dt) == 0:
        print(f"[MOT] No predictions in {tSource}; "
              "all GT objects will be counted as misses.")

    max_d2 = max_dist_m ** 2
    accs = []
    for seq in np.unique(gt[:, 0]).astype(int):
        acc = mm.MOTAccumulator()
        for frame in np.unique(gt[:, 1]).astype(int):
            gt_dets = gt[np.logical_and(gt[:, 0] == seq, gt[:, 1] == frame)][:, (2, 8, 9)]
            dt_dets = dt[np.logical_and(dt[:, 0] == seq, dt[:, 1] == frame)][:, (2, 8, 9)]

            # format: gt, t.  norm2squared_matrix keeps distances
            # <= max_d2, so a centre exactly 100 cm away is matchable.
            C = mm.distances.norm2squared_matrix(
                gt_dets[:, 1:3] * scale,
                dt_dets[:, 1:3] * scale,
                max_d2=max_d2,
            )
            C = np.sqrt(C)

            acc.update(gt_dets[:, 0].astype('int').tolist(),
                       dt_dets[:, 0].astype('int').tolist(),
                       C,
                       frameid=frame)
        accs.append(acc)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True,
    )
    print("\n")
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    return summary
