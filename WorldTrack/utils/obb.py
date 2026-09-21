# WorldTrack/utils/obb.py
"""
Shared oriented-bounding-box geometry and BEV <-> world coordinate helpers.

This module is a pure refactor: every function here was previously
implemented inline, either as a private function in ``models/loss.py``
(``_make_box_corners``) or duplicated three times inside
``world_track.py`` (mem-grid -> world-grid -> world-centimetres).

The arithmetic below is intentionally written in exactly the same form as
the original inline code (explicit multiply-add, same operand order, same
dtypes, no matmul) so that every loss value, metric and exported file is
unchanged down to the last bit.
"""

import numpy as np
import torch

# utils/decode.py imports only torch, so this cannot create a circular
# import (models/loss.py -> utils.obb -> utils.decode -> torch).
from utils import decode as _decode


# ──────────────────────────────────────────────────────────────────
# 3D box geometry
# ──────────────────────────────────────────────────────────────────
def make_box_corners(cx, cy, z_base, l, w, h, sin_y, cos_y, device):
    """Build 8 oriented-box corners in world coords (cm).

    ``sin_y`` / ``cos_y`` are passed in directly (already normalised) so
    we never take atan2() of a possibly-zero (sin, cos) pair — the
    derivative of atan2 is cos/(sin^2+cos^2), which explodes at the
    origin and produced the intermittent gradient spikes that starved
    the rest of the network through global gradient clipping.

    Corner order (sign of x, y, z-offset), preserved exactly:
        0:(+,+,0) 1:(+,-,0) 2:(-,-,0) 3:(-,+,0)
        4:(+,+,h) 5:(+,-,h) 6:(-,-,h) 7:(-,+,h)

    Returns
    -------
    (8, 3) tensor of [x_cm, y_cm, z_cm]
    """
    dx = l / 2.0
    dy = w / 2.0

    signs = torch.tensor(
        [[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0],
         [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]],
        dtype=torch.float32, device=device,
    )

    xs = signs[:, 0] * dx
    ys = signs[:, 1] * dy
    zs = z_base + signs[:, 2] * h

    xr = cos_y * xs - sin_y * ys + cx
    yr = sin_y * xs + cos_y * ys + cy

    return torch.stack([xr, yr, zs], dim=1)  # (8, 3)


# Backwards-compatible private alias, so that any existing
# ``from models.loss import _make_box_corners`` (or a direct
# ``utils.obb._make_box_corners``) keeps resolving.
_make_box_corners = make_box_corners


# ──────────────────────────────────────────────────────────────────
# Coordinate transforms:  BEV memory grid -> world grid -> world cm
# ──────────────────────────────────────────────────────────────────
def mem_to_ref_xy(vox_util, xy_mem, Y, Z, X, assert_cube=False):
    """BEV *memory* coordinates -> *reference* (world-grid) coordinates.

    Parameters
    ----------
    vox_util : utils.vox.VoxelUtil
    xy_mem   : (..., N, 2) or (..., N, 3) tensor of BEV memory coords.
               If only (x, y) are given, a zero z-column is appended
               (exactly what the previous inline code did).
    Y, Z, X  : voxel-grid resolution.

    Returns
    -------
    (..., N, 2) tensor of world-grid (x, y).
    """
    if xy_mem.shape[-1] == 2:
        xyz_mem = torch.cat(
            (xy_mem, torch.zeros_like(xy_mem[..., 0:1])), dim=-1
        )
    else:
        xyz_mem = xy_mem

    ref = vox_util.Mem2Ref(xyz_mem, Y, Z, X, assert_cube=assert_cube)
    return ref[..., :2]


def worldgrid_to_worldcm(grid_xy, worldcoord_from_worldgrid):
    """World-grid (cell) coordinates -> world coordinates in centimetres.

    ``worldcoord_from_worldgrid`` is the 3x3 affine matrix produced by the
    dataset (``base.worldcoord_from_worldgrid_mat``):

        world_x = W[0,0]*grid_x + W[0,1]*grid_y + W[0,2]
        world_y = W[1,0]*grid_x + W[1,1]*grid_y + W[1,2]

    Works transparently with ``torch.Tensor`` and ``np.ndarray`` inputs
    and preserves the input dtype / device. Accepts any leading shape,
    including a bare (2,) single point and an empty (0, 2) array.
    """
    W = worldcoord_from_worldgrid

    gx = grid_xy[..., 0]
    gy = grid_xy[..., 1]

    wx = W[0, 0] * gx + W[0, 1] * gy + W[0, 2]
    wy = W[1, 0] * gx + W[1, 1] * gy + W[1, 2]

    if isinstance(grid_xy, torch.Tensor):
        return torch.stack((wx, wy), dim=-1)
    return np.stack((wx, wy), axis=-1)


def mem_to_worldcm(vox_util, xy_mem, Y, Z, X,
                   worldcoord_from_worldgrid, assert_cube=False):
    """Convenience composition: BEV memory coords -> world centimetres.

    Equivalent to ``worldgrid_to_worldcm(mem_to_ref_xy(...), W)``.
    """
    ref_xy = mem_to_ref_xy(vox_util, xy_mem, Y, Z, X,
                           assert_cube=assert_cube)
    return worldgrid_to_worldcm(ref_xy, worldcoord_from_worldgrid)

# ──────────────────────────────────────────────────────────────────
# canonical dense-BEV -> oriented-bounding-box decoding
# ──────────────────────────────────────────────────────────────────
# Single source of truth for turning the model's dense BEV heads into a
# flat list of records. Two mutually exclusive schemas:
#
#   has_3d_gt=True  -> full OBB record (11 fields)
#   has_3d_gt=False -> point-only record (4 fields), i.e. exactly the
#                      information the current 2D-only path produces.
#

#: multiplier used to fold (sequence_num, frame) into one unique id,
#: identical to the convention already used in world_track.py.
GLOBAL_FRAME_STRIDE = 1_000_000

OBB_FIELDS_3D = (
    'frame_id', 'x_cm', 'y_cm', 'z_base_cm', 'yaw_rad',
    'length_cm', 'width_cm', 'height_cm',
    'posture_class', 'posture_prob', 'score',
)

OBB_FIELDS_2D = ('frame_id', 'x_cm', 'y_cm', 'score')


def obb_fields(has_3d_gt):
    """Return the tuple of field names produced by ``decode_obb``."""
    return OBB_FIELDS_3D if has_3d_gt else OBB_FIELDS_2D


@torch.no_grad()
def decode_obb(
    output,
    item,
    vox_util,
    conf_threshold,
    has_3d_gt,
    Y, Z, X,
    size_scale_cm=100.0,
    max_detections=60,
    z_base_cm=0.0,
    coord_space='world_cm',
    use_global_frame_id=True,
    return_arrays=False,
    # ── duplicate suppression (forwarded to decode.decoder) ──
    nms_kernel=3,
    nms_radius=0.0,
    nms_iou=0.0,
    nms_cell_size_cm=None,
    nms_score_threshold=0.0,
    # callable (B,K,2)->attr dict; REQUIRED when has_3d_gt=True
    attr_fn=None,
):
    """Decode dense BEV head outputs into a list of detection records.

    Follows the project-wide convention: decode Top-K unconditionally,
    then keep ``score > conf_threshold`` (same as ``test_step`` and
    ``_accumulate_val_moda``).

    Parameters
    ----------
    output : dict
        Model output. Must contain ``instance_center`` (raw logits) and
        ``instance_offset``. When ``has_3d_gt`` is True it must also
        contain ``instance_yaw``, ``instance_size`` (METRES) and
        ``instance_posture`` (logit).
    item : dict or None
        Batch input. Used for ``frame``, ``sequence_num`` and
        ``worldcoord_from_worldgrid`` ((3,3) or (B,3,3)). May be None
        only when ``coord_space='grid'``.
    vox_util : utils.vox.VoxelUtil
    conf_threshold : float
    has_3d_gt : bool
        False -> point-only decoding (no 3D head arguments are passed to
        ``decode.decoder`` at all, so the result is identical to the
        existing 2D path). True -> full OBB decoding.
    Y, Z, X : int
        BEV/voxel resolution.
    size_scale_cm : float
        metres -> centimetres factor handed to ``decode.decoder``.
    max_detections : int
        Top-K.
    z_base_cm : float
        Constant ground height written into the record (the network has
        no z head; GT ``z_base`` is 0.0 in ``grid_gt_3d``).
    coord_space : {'world_cm', 'grid'}
        'world_cm' (default) -> x/y in centimetres.
        'grid'               -> x/y in world-grid cells, i.e. bit-identical
                                to the values pushed into
                                ``moda_pred_list`` today.
    use_global_frame_id : bool
        True -> frame_id = sequence_num * 1e6 + frame (matches
        ``test_step``). Degenerates to the raw frame when
        ``sequence_num`` is absent/zero.
    return_arrays : bool
        True -> return an ``(N, F)`` float64 numpy array instead of a
        list of tuples.

    Returns
    -------
    list[tuple] (or np.ndarray if ``return_arrays``)
        Records ordered by descending score within each batch element,
        batch elements in order. Field layout = ``obb_fields(has_3d_gt)``.
    """
    if coord_space not in ('world_cm', 'grid'):
        raise ValueError(
            f"coord_space must be 'world_cm' or 'grid', got {coord_space!r}"
        )

    center_e = output['instance_center']
    offset_e = output['instance_offset']
    B = int(center_e.shape[0])

    # decode
    if has_3d_gt:
        if attr_fn is None:
            raise ValueError(
                "has_3d_gt=True but attr_fn is None -- pass the model's "
                "per-object attribute head, e.g. "
                "WorldTrackModel._attr_fn_3d(output)."
            )
        decoded = _decode.decoder(
            center_e.sigmoid(), offset_e, None,
            K=max_detections,
            attr_fn=attr_fn,
            nms_kernel=nms_kernel,
            nms_radius=nms_radius,
            nms_iou=nms_iou,
            nms_cell_size_cm=nms_cell_size_cm,
            nms_score_threshold=nms_score_threshold,
        )
        xy_e, _xy_prev, scores_e, _clses, extra = decoded
    else:
        # point-only call (no 3D kwargs) -> radius NMS only.
        # decode.decoder() returns a plain 4-tuple here because no
        # yaw_e/size3d_e/posture_e are passed (extra dict is empty).
        decoded = _decode.decoder(
            center_e.sigmoid(), offset_e, None,
            K=max_detections,
            nms_kernel=nms_kernel,
            nms_radius=nms_radius,
            nms_score_threshold=nms_score_threshold,
        )
        xy_e, _xy_prev, scores_e, _clses = decoded
        extra = {}

    # positions: BEV memory -> world-grid -> (optionally) cm
    ref_xy = mem_to_ref_xy(vox_util, xy_e.float(), Y, Z, X).cpu()  # (B,K,2)

    if coord_space == 'grid':
        pos_xy = ref_xy
    else:
        if item is None or 'worldcoord_from_worldgrid' not in item:
            raise ValueError(
                "coord_space='world_cm' needs "
                "item['worldcoord_from_worldgrid']"
            )
        Wm = item['worldcoord_from_worldgrid'].float().cpu()
        if Wm.dim() == 2:
            Wm = Wm.unsqueeze(0).expand(B, -1, -1)
        pos_xy = torch.stack(
            [worldgrid_to_worldcm(ref_xy[b], Wm[b]) for b in range(B)],
            dim=0,
        )

    # frame ids
    if item is not None and 'frame' in item:
        frames = item['frame'].detach().cpu().reshape(-1).long()
    else:
        frames = torch.zeros(B, dtype=torch.long)
    if (use_global_frame_id and item is not None
            and 'sequence_num' in item):
        seqs = item['sequence_num'].detach().cpu().reshape(-1).long()
    else:
        seqs = torch.zeros(B, dtype=torch.long)
    frame_ids = (seqs * GLOBAL_FRAME_STRIDE + frames).tolist()

    # build records
    scores_cpu = scores_e.float().cpu()
    if has_3d_gt:
        yaw_cpu = extra['yaw_angle'].float().cpu()          # (B,K)
        dim_cpu = extra['dimensions'].float().cpu()          # (B,K,3) cm
        pprob_cpu = extra['posture_prob'].float().cpu()      # (B,K)
        pcls_cpu = extra['posture_class'].long().cpu()       # (B,K)

    records = []
    for b in range(B):
        keep = (scores_cpu[b] > conf_threshold).nonzero(
            as_tuple=True)[0].tolist()
        fid = int(frame_ids[b])
        for k in keep:  # ascending k == descending score (topk order)
            x = float(pos_xy[b, k, 0])
            y = float(pos_xy[b, k, 1])
            s = float(scores_cpu[b, k])
            if not has_3d_gt:
                records.append((fid, x, y, s))
                continue
            records.append((
                fid, x, y, float(z_base_cm),
                float(yaw_cpu[b, k]),
                float(dim_cpu[b, k, 0]),
                float(dim_cpu[b, k, 1]),
                float(dim_cpu[b, k, 2]),
                int(pcls_cpu[b, k]),
                float(pprob_cpu[b, k]),
                s,
            ))

    if return_arrays:
        n_fields = len(obb_fields(has_3d_gt))
        if not records:
            return np.zeros((0, n_fields), dtype=np.float64)
        return np.asarray(records, dtype=np.float64).reshape(-1, n_fields)
    return records
