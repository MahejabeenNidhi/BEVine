import torch
import torch.nn as nn
import torch.nn.functional as F

import math as _math
from collections import defaultdict

from utils import basic


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid):
        loss = self.loss_fn(ypred, ytgt)
        loss = basic.reduce_masked_mean(loss, valid)
        return loss


class FocalLoss(torch.nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self, use_distance_weight=False):
        super(FocalLoss, self).__init__()
        self.use_distance_weight = use_distance_weight

    def forward(self, pred, gt):
        """ Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
                pred (batch x c x h x w)
                gt_regr (batch x c x h x w)
        """
        # find pos indices and neg indices
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        distance_weight = torch.ones_like(gt)
        if self.use_distance_weight:
            w, h = gt.shape[-2:]
            xs = torch.linspace(-1, 1, steps=h, device=gt.device)
            ys = torch.linspace(-1, 1, steps=w, device=gt.device)
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            distance_weight = 9 * torch.sin(torch.sqrt(x * x + y * y)) + 1

        # following paper alpha 2, beta 4
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * distance_weight
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds * distance_weight

        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


def balanced_mse_loss(pred, gt, valid=None):
    pos_mask = gt.gt(0.5).float()
    neg_mask = gt.lt(0.5).float()
    if valid is None:
        valid = torch.ones_like(pos_mask)
    mse_loss = F.mse_loss(pred, gt, reduction='none')
    pos_loss = basic.reduce_masked_mean(mse_loss, pos_mask * valid)
    neg_loss = basic.reduce_masked_mean(mse_loss, neg_mask * valid)
    loss = (pos_loss + neg_loss) * 0.5

    return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, rotbin, rotres):
        loss = compute_rot_loss(output, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='mean')


def compute_bin_loss(output, target, mask):
    # mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res

def smooth_l1_reprojection(pred: torch.Tensor, gt: torch.Tensor
                           ) -> torch.Tensor:
    """Per-match smooth-L1 loss. Returns (M,) scalar losses."""
    return F.smooth_l1_loss(pred, gt, reduction='none').sum(dim=-1)


def cauchy_reprojection(pred: torch.Tensor, gt: torch.Tensor,
                        c: float = 50.0) -> torch.Tensor:
    """Cauchy kernel: c²·log(1 + ||r||²/c²)/2. Returns (M,)."""
    r_sq = ((pred - gt) ** 2).sum(dim=-1)
    c_sq = c * c
    return c_sq * torch.log1p(r_sq / c_sq) / 2.0


def gmc_reprojection(pred: torch.Tensor, gt: torch.Tensor,
                     c: float = 50.0) -> torch.Tensor:
    """Geman-McClure kernel: ||r||²/(||r||²+c²). Returns (M,)."""
    r_sq = ((pred - gt) ** 2).sum(dim=-1)
    c_sq = c * c
    return r_sq / (r_sq + c_sq)


_REPROJ_KERNELS = {
    'smooth_l1': smooth_l1_reprojection,
    'cauchy':    cauchy_reprojection,
    'gmc':       gmc_reprojection,
}

def _compute_visibility_counts(grid_gt, img_gt_2d):
    """
    Returns dict {(batch_idx, pid_int): n_cameras_seeing_this_animal}.
    """
    B, S = img_gt_2d.shape[:2]
    counts = {}
    for b in range(B):
        valid = grid_gt[b, :, 2] > 0
        if not valid.any():
            continue
        pid_set = set(grid_gt[b, valid, 2].long().tolist())
        for pid_val in pid_set:
            n_vis = 0
            for c_idx in range(S):
                cam_gt = img_gt_2d[b, c_idx]
                det_pids = cam_gt[:, 4].long()
                bbox_vis = ~(
                    (cam_gt[:, 0] == -1) & (cam_gt[:, 1] == -1) &
                    (cam_gt[:, 2] == -1) & (cam_gt[:, 3] == -1)
                )
                if ((det_pids == pid_val) & bbox_vis).any():
                    n_vis += 1
            counts[(b, pid_val)] = n_vis
    return counts

def reprojection_loss(
    grid_gt,
    img_gt_2d,
    intrinsic_original,
    refined_extrinsics,
    worldcoord_from_worldgrid_mat,
    return_per_cam_stats: bool = False,
    camera_weighting: str = 'proportional',   # 'proportional' or 'equal'
    min_obs_per_camera: int = 0,
    soft_gate: bool = False,
    soft_gate_temp: float = 1.0,
    loss_type: str = 'smooth_l1',
    robust_scale: float = 50.0,
    use_confidence_weights: bool = False,
    depth_weight_clip: tuple = (0.1, 1.0),
):
    """
    Reprojection error with configurable camera weighting.

    camera_weighting:
      'proportional' — each match gets equal weight regardless of which
                       camera it came from (original behavior). A camera
                       with 20 matches naturally contributes 10× more
                       than one with 2.  DEFAULT for backward compat.
      'equal'        — each camera's mean error contributes equally.
                       ONLY use this with min_obs_per_camera >= 3 to
                       avoid amplifying noisy low-observation cameras.
    """
    B = grid_gt.shape[0]
    S = img_gt_2d.shape[1]
    device = grid_gt.device

    W = worldcoord_from_worldgrid_mat
    if W.dim() == 2:
        W = W.unsqueeze(0).expand(B, -1, -1)

    kernel_fn = _REPROJ_KERNELS.get(loss_type)
    if kernel_fn is None:
        raise ValueError(
            f"Unknown reproj loss type '{loss_type}'. "
            f"Choose from {list(_REPROJ_KERNELS.keys())}"
        )

    vis_counts = (
        _compute_visibility_counts(grid_gt, img_gt_2d)
        if use_confidence_weights else {}
    )

    # Accumulators
    cam_loss_accum: dict = defaultdict(list)       # for 'equal' mode
    all_per_match_losses: list = []                 # for 'proportional' mode
    cam_n_matches: dict = defaultdict(int)          # diagnostics
    cam_residual_sum: dict = defaultdict(float)     # diagnostics

    for b in range(B):
        valid_mask = grid_gt[b, :, 2] > 0
        if not valid_mask.any():
            continue

        grid_coords = grid_gt[b][valid_mask]
        grid_x = grid_coords[:, 0].float()
        grid_y = grid_coords[:, 1].float()
        pids = grid_coords[:, 2].long()
        N = pids.shape[0]

        Wb = W[b]
        world_x = Wb[0, 0] * grid_x + Wb[0, 1] * grid_y + Wb[0, 2]
        world_y = Wb[1, 0] * grid_x + Wb[1, 1] * grid_y + Wb[1, 2]
        world_pts_h = torch.stack([
            world_x, world_y,
            torch.zeros(N, device=device, dtype=world_x.dtype),
            torch.ones(N, device=device, dtype=world_x.dtype),
        ], dim=1)

        for c in range(S):
            E = refined_extrinsics[b, c]
            K = intrinsic_original[b, c]

            cam_pts = E[:3, :] @ world_pts_h.T
            cam_z_orig = cam_pts[2, :]
            in_front = cam_z_orig > 0
            cam_z = cam_z_orig.clamp(min=1e-4)

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            u_pred = fx * cam_pts[0, :] / cam_z + cx
            v_pred = fy * cam_pts[1, :] / cam_z + cy

            img_gt_cam = img_gt_2d[b, c]
            det_pids = img_gt_cam[:, 4].long()
            bbox_visible = ~(
                (img_gt_cam[:, 0] == -1) &
                (img_gt_cam[:, 1] == -1) &
                (img_gt_cam[:, 2] == -1) &
                (img_gt_cam[:, 3] == -1)
            )
            det_valid = (det_pids > 0) & bbox_visible
            if not det_valid.any():
                continue

            valid_dets = img_gt_cam[det_valid]
            valid_det_pids = det_pids[det_valid]
            det_foot_u = (valid_dets[:, 0] + valid_dets[:, 2]) / 2.0
            det_foot_v = valid_dets[:, 3]

            if use_confidence_weights:
                depths_in_front = cam_z_orig[in_front]
                if depths_in_front.numel() > 1:
                    depth_mean_t = depths_in_front.mean().detach()
                    depth_std_t = depths_in_front.std().clamp(min=1.0).detach()
                elif depths_in_front.numel() == 1:
                    depth_mean_t = depths_in_front[0].detach()
                    depth_std_t = torch.tensor(5.0, device=device)
                else:
                    depth_mean_t = torch.tensor(5.0, device=device)
                    depth_std_t = torch.tensor(5.0, device=device)

            # Collect matches
            match_pred_list = []
            match_gt_list = []
            match_weight_list = []

            for i in range(N):
                if not in_front[i]:
                    continue
                matches_idx = (
                    valid_det_pids == pids[i]
                ).nonzero(as_tuple=True)[0]
                if len(matches_idx) == 0:
                    continue
                j = matches_idx[0]

                match_pred_list.append(
                    torch.stack([u_pred[i], v_pred[i]])
                )
                match_gt_list.append(
                    torch.stack([det_foot_u[j], det_foot_v[j]])
                )

                if use_confidence_weights:
                    z_i = cam_z_orig[i].detach()
                    w_depth = torch.sigmoid(
                        -(z_i - depth_mean_t) / depth_std_t
                    ).clamp(min=depth_weight_clip[0], max=depth_weight_clip[1])
                    det_j = valid_dets[j]
                    h_bb = (det_j[3] - det_j[1]).detach().clamp(min=1.0)
                    w_bb = (det_j[2] - det_j[0]).detach().clamp(min=1.0)
                    w_aspect = torch.clamp(h_bb / w_bb, max=1.0)
                    pid_val = pids[i].item()
                    n_vis = vis_counts.get((b, pid_val), 1)
                    w_vis = _math.log(1.0 + n_vis) / _math.log(1.0 + S)
                    match_weight_list.append(w_depth * w_aspect * w_vis)
                else:
                    match_weight_list.append(
                        torch.tensor(1.0, device=device)
                    )

            n_matches = len(match_pred_list)
            if n_matches == 0:
                continue

            # observation gate
            if min_obs_per_camera > 0:
                if soft_gate:
                    gate_w = torch.sigmoid(
                        (torch.tensor(float(n_matches), device=device)
                         - float(min_obs_per_camera)) / soft_gate_temp
                    )
                else:
                    if n_matches < min_obs_per_camera:
                        if return_per_cam_stats:
                            with torch.no_grad():
                                _p = torch.stack(match_pred_list)
                                _g = torch.stack(match_gt_list)
                                _r = (_p - _g).norm(dim=1).mean().item()
                                cam_n_matches[c] += n_matches
                                cam_residual_sum[c] += _r * n_matches
                        continue
                    gate_w = 1.0
            else:
                gate_w = 1.0

            pred_uv = torch.stack(match_pred_list)
            gt_uv = torch.stack(match_gt_list)

            # robust kernel
            if loss_type == 'smooth_l1':
                per_match = kernel_fn(pred_uv, gt_uv)
            else:
                per_match = kernel_fn(pred_uv, gt_uv, c=robust_scale)

            # weighted mean
            if use_confidence_weights:
                weights = torch.stack(match_weight_list).detach()
                cam_mean = (
                    (per_match * weights).sum()
                    / weights.sum().clamp(min=1e-6)
                ) * gate_w
                # For proportional mode: store individually weighted losses
                all_per_match_losses.append(
                    per_match * weights / weights.sum().clamp(min=1e-6)
                    * gate_w * n_matches
                )
            else:
                cam_mean = per_match.mean() * gate_w
                # For proportional mode: store per-match losses with gate
                all_per_match_losses.append(per_match * gate_w)

            cam_loss_accum[c].append(cam_mean)

            # Diagnostics
            with torch.no_grad():
                res_px = (pred_uv - gt_uv).norm(dim=1).mean().item()
            cam_n_matches[c] += n_matches
            cam_residual_sum[c] += res_px * n_matches

    # ── Aggregation: choose weighting strategy ────────────
    if camera_weighting == 'equal':
        # I1: equal weight per camera
        cam_means = []
        for c_key in sorted(cam_loss_accum.keys()):
            entries = cam_loss_accum[c_key]
            if entries:
                cam_means.append(torch.stack(entries).mean())
        if cam_means:
            loss = torch.stack(cam_means).mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

    elif camera_weighting == 'proportional':
        # Each match gets equal weight; cameras with more matches
        # naturally contribute more (original behavior).
        if all_per_match_losses:
            loss = torch.cat(all_per_match_losses).mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

    else:
        raise ValueError(
            f"Unknown camera_weighting '{camera_weighting}'. "
            f"Choose 'proportional' or 'equal'."
        )

    if return_per_cam_stats:
        stats = {}
        for c_idx in range(S):
            n = cam_n_matches.get(c_idx, 0)
            mean_res = cam_residual_sum.get(c_idx, 0.0) / max(n, 1)
            stats[c_idx] = (n, mean_res)
        return loss, stats

    return loss