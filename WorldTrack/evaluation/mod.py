import os
import numpy as np
# from CLEAR_MOD_HUN import CLEAR_MOD_HUN
from evaluation.CLEAR_MOD_HUN import CLEAR_MOD_HUN

def _load_mod_file(path, required, label):
    """Load a whitespace-delimited MOD file as a guaranteed 2-D array."""
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"Required {label} MOD file does not exist: {path}"
            )
        return np.empty((0, 3), dtype=float)

    if os.path.getsize(path) == 0:
        if required:
            raise ValueError(
                f"Required {label} MOD file is empty: {path}"
            )
        return np.empty((0, 3), dtype=float)

    arr = np.loadtxt(path, ndmin=2)

    if arr.size == 0:
        if required:
            raise ValueError(
                f"Required {label} MOD file contains no rows: {path}"
            )
        return np.empty((0, 3), dtype=float)

    if arr.shape[1] < 3:
        raise ValueError(
            f"{label.capitalize()} MOD file has {arr.shape[1]} column(s), "
            f"but at least 3 are required: {path}"
        )

    return arr

def modMetricsCalculator(res_fpath, gt_fpath, td_cells=10.0):
    # td_cells: association gate in GRID CELLS.
    # 10.0 cells = 100 cm on the 10 cm mmCows grid (matches MOTA's 100 cm).

    gtRaw = _load_mod_file(
        gt_fpath, required=True, label='ground-truth'
    )
    detRaw = _load_mod_file(
        res_fpath, required=False, label='prediction'
    )

    if detRaw.shape[0] == 0:
        print(f"[MOD] No predictions in {res_fpath}; "
              "returning zero detection metrics.")
        MODP, MODA, recall, precision = 0, 0, 0, 0
        return MODP, MODA, recall, precision

    frames = np.unique(detRaw[:, 0])
    frame_ctr = 0
    gt_flag = True
    det_flag = True

    gtAllMatrix = 0
    detAllMatrix = 0

    for t in frames:
        idxs = np.where(gtRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)
        tmp_arr = np.zeros(shape=(idx_len, 4))
        tmp_arr[:, 0] = np.array([frame_ctr for n in range(idx_len)])
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])
        tmp_arr[:, 2] = np.array([j for j in gtRaw[idx, 1]])
        tmp_arr[:, 3] = np.array([k for k in gtRaw[idx, 2]])

        if gt_flag:
            gtAllMatrix = tmp_arr
            gt_flag = False
        else:
            gtAllMatrix = np.concatenate((gtAllMatrix, tmp_arr), axis=0)
        idxs = np.where(detRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)
        tmp_arr = np.zeros(shape=(idx_len, 4))
        tmp_arr[:, 0] = np.array([frame_ctr for n in range(idx_len)])
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])
        tmp_arr[:, 2] = np.array([j for j in detRaw[idx, 1]])
        tmp_arr[:, 3] = np.array([k for k in detRaw[idx, 2]])

        if det_flag:
            detAllMatrix = tmp_arr
            det_flag = False
        else:
            detAllMatrix = np.concatenate((detAllMatrix, tmp_arr), axis=0)
        frame_ctr += 1
    recall, precision, MODA, MODP = CLEAR_MOD_HUN(
        gtAllMatrix, detAllMatrix, td=td_cells)
    return recall, precision, MODA, MODP
