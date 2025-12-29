import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import basic


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([pos_weight]),
            reduction='none'
        )

    def forward(self, ypred, ytgt, valid):
        loss = self.loss_fn(ypred, ytgt)
        loss = basic.reduce_masked_mean(loss, valid)
        return loss


class FocalLoss(torch.nn.Module):
    '''
    Vectorized per-sample normalized Focal Loss for multi-dataset training.

    Key improvements:
    - Per-sample normalization (resolution-invariant)
    - Numerically stable (gradient clipping + epsilon)
    - Vectorized (no Python loops)
    '''

    def __init__(self, use_distance_weight=False):
        super(FocalLoss, self).__init__()
        self.use_distance_weight = use_distance_weight

    def forward(self, pred, gt):
        """
        Args:
            pred: (B, C, H, W) - predicted heatmap (after sigmoid)
            gt: (B, C, H, W) - ground truth heatmap
        Returns:
            Scalar loss (averaged over batch)
        """
        # ===== NUMERICAL STABILITY =====
        pred = pred.clamp(min=1e-6, max=1 - 1e-6)

        # ===== POSITIVE/NEGATIVE MASKS =====
        pos_inds = gt.eq(1).float()  # (B, C, H, W)
        neg_inds = gt.lt(1).float()

        # ===== OPTIONAL DISTANCE WEIGHTING =====
        distance_weight = torch.ones_like(gt)
        if self.use_distance_weight:
            h, w = gt.shape[-2:]
            ys = torch.linspace(-1, 1, steps=h, device=gt.device)
            xs = torch.linspace(-1, 1, steps=w, device=gt.device)
            y, x = torch.meshgrid(ys, xs, indexing='ij')
            dist = torch.sqrt(x * x + y * y)
            distance_weight = 9 * torch.sin(dist) + 1
            distance_weight = distance_weight.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # ===== FOCAL LOSS PARAMETERS =====
        alpha = 2  # Focusing parameter
        beta = 4  # Negative modulating factor
        neg_weights = torch.pow(1 - gt, beta)

        # ===== PER-PIXEL LOSSES (with numerical stability) =====
        # Use log(x + eps) instead of clamp + log for better gradients
        eps = 1e-8
        pos_loss_perpixel = (
                torch.log(pred + eps) *
                torch.pow(1 - pred, alpha) *
                pos_inds *
                distance_weight
        )
        neg_loss_perpixel = (
                torch.log(1 - pred + eps) *
                torch.pow(pred, alpha) *
                neg_weights *
                neg_inds *
                distance_weight
        )

        # ===== VECTORIZED PER-SAMPLE NORMALIZATION =====
        # Sum over (C, H, W) dimensions, keeping batch dimension
        num_pos_per_sample = pos_inds.sum(dim=[1, 2, 3]).clamp(min=1.0)  # (B,)
        pos_loss_per_sample = pos_loss_perpixel.sum(dim=[1, 2, 3])  # (B,)
        neg_loss_per_sample = neg_loss_perpixel.sum(dim=[1, 2, 3])  # (B,)

        # Normalize each sample by its own number of positive pixels
        loss_per_sample = -(pos_loss_per_sample + neg_loss_per_sample) / num_pos_per_sample  # (B,)

        # Average across batch (each sample contributes equally)
        loss = loss_per_sample.mean()

        return loss


def balanced_mse_loss(pred, gt, valid=None):
    """
    Balanced MSE loss for regression tasks.

    Separately normalizes positive and negative examples to handle class imbalance.
    """
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
    """Wrapper for binned rotation loss"""

    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, rotbin, rotres):
        loss = compute_rot_loss(output, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    """
    Smooth L1 loss for continuous rotation residuals.

    Args:
        output: (N,) - predicted sin/cos values
        target: (N,) - ground truth sin/cos values

    Returns:
        Scalar loss
    """
    return F.smooth_l1_loss(output, target, reduction='mean')


def compute_bin_loss(output, target, mask):
    """
    Masked cross-entropy loss for rotation bin classification.

    Args:
        output: (N, 2) - logits for binary classification
        target: (N,) - class labels (0 or 1)
        mask: (N, 1) - binary mask (1 = valid, 0 = ignore)

    Returns:
        Scalar loss averaged over valid elements only
    """
    # Compute loss for all elements
    loss_per_element = F.cross_entropy(output, target, reduction='none')  # (N,)

    # Apply mask
    mask = mask.squeeze(-1)  # (N, 1) -> (N,)
    masked_loss = loss_per_element * mask

    # Average over valid elements only (prevents division by zero)
    num_valid = mask.sum().clamp(min=1)
    loss = masked_loss.sum() / num_valid

    return loss


def compute_rot_loss(output, target_bin, target_res, mask):
    """
    Multi-bin rotation loss (as in CenterNet).

    Rotation is represented as:
    - 2 bins (each covering 180°)
    - Each bin predicts: [bin_confidence, sin(θ), cos(θ)]

    Args:
        output: (B, H, W, 8) - [bin1_cls[0:2], bin1_sin, bin1_cos,
                                  bin2_cls[4:6], bin2_sin, bin2_cos]
        target_bin: (B, H, W, 2) - [bin1_cls, bin2_cls] (which bin is active)
        target_res: (B, H, W, 2) - [bin1_residual, bin2_residual] (angle residuals)
        mask: (B, H, W, 1) - valid pixel mask

    Returns:
        Scalar loss
    """
    # Flatten spatial dimensions
    output = output.view(-1, 8)  # (N, 8)
    target_bin = target_bin.view(-1, 2)  # (N, 2)
    target_res = target_res.view(-1, 2)  # (N, 2)
    mask = mask.view(-1, 1)  # (N, 1)

    # ===== BIN CLASSIFICATION LOSSES =====
    # Which bin is the correct one?
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)

    # ===== RESIDUAL REGRESSION LOSSES =====
    # Only compute residual loss for the active bin
    loss_res = torch.zeros_like(loss_bin1)

    # Bin 1 residuals (sin/cos)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())

        loss_sin1 = compute_res_loss(
            valid_output1[:, 2],
            torch.sin(valid_target_res1[:, 0])
        )
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3],
            torch.cos(valid_target_res1[:, 0])
        )
        loss_res += loss_sin1 + loss_cos1

    # Bin 2 residuals (sin/cos)
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())

        loss_sin2 = compute_res_loss(
            valid_output2[:, 6],
            torch.sin(valid_target_res2[:, 1])
        )
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7],
            torch.cos(valid_target_res2[:, 1])
        )
        loss_res += loss_sin2 + loss_cos2

    return loss_bin1 + loss_bin2 + loss_res