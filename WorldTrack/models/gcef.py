"""
Enhanced Global Context Fusion (EGCF) Module
Based on: Hu et al. 2025

Key principle: Refine LOCAL BEV features using GLOBAL context

Operates on 2D BEV features (not 3D voxels)
Applied AFTER camera aggregation, BEFORE Z-flattening (for Segnet)
Applied AFTER perspective warping (for MVDet)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalContextEnhancedFusion(nn.Module):
    """
    GCEF Module: Refines local features using global context (2D version).

    Args:
        feat_dim: Feature dimension (e.g., 128)
        reduction_ratio: Channel reduction for attention (default: 16)
    """

    def __init__(self, feat_dim=128, reduction_ratio=16):
        super().__init__()
        self.feat_dim = feat_dim

        self.static_fusion = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_dim, feat_dim // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // reduction_ratio, feat_dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, local_feat, global_feat):
        assert len(local_feat.shape) == 4, f"Expected 4D tensor, got {local_feat.shape}"
        concat_feat = torch.cat([global_feat, local_feat], dim=1)
        fused_feat = self.static_fusion(concat_feat)
        attention_weights = self.attention(fused_feat)
        refined_feat = fused_feat * attention_weights
        return refined_feat


class GCEF3D(nn.Module):
    """
    3D version of GCEF for voxel features.
    """

    def __init__(self, feat_dim=128, reduction_ratio=16):
        super().__init__()
        self.feat_dim = feat_dim

        self.static_fusion = nn.Sequential(
            nn.Conv3d(feat_dim * 2, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(feat_dim),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(feat_dim, feat_dim // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(feat_dim // reduction_ratio, feat_dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, local_feat, global_feat):
        assert len(local_feat.shape) == 5, f"Expected 5D tensor, got {local_feat.shape}"
        concat_feat = torch.cat([global_feat, local_feat], dim=1)
        fused_feat = self.static_fusion(concat_feat)
        attention_weights = self.attention(fused_feat)
        return fused_feat * attention_weights


class GCEFIntegrationSegnet(nn.Module):
    """
    Integrates GCEF into Segnet’s 3D voxel pipeline.
    """

    def __init__(self, feat_dim=128, num_cameras=None, use_camera_compressor=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_cameras = num_cameras
        self.use_camera_compressor = use_camera_compressor

        self.gcef = GCEF3D(feat_dim=feat_dim)

        if num_cameras is not None and use_camera_compressor:
            self.cam_compressor = nn.Sequential(
                nn.Conv3d(feat_dim * num_cameras, feat_dim,
                          kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm3d(feat_dim),
                nn.ReLU(inplace=True),
                nn.Conv3d(feat_dim, feat_dim, kernel_size=1, bias=False),
            )
        else:
            self.cam_compressor = None

    def forward(self, feat_mems, mask_mems=None):
        if len(feat_mems.shape) == 6:
            B, S, C, Z, Y, X = feat_mems.shape
            has_multi_cameras = True
        else:
            B, C, Z, Y, X = feat_mems.shape
            S = 1
            has_multi_cameras = False
            feat_mems = feat_mems.unsqueeze(1)
            if mask_mems is not None:
                mask_mems = mask_mems.unsqueeze(1)

        if has_multi_cameras and S > 1:
            if mask_mems is not None:
                global_bev = self._masked_mean_aggregate(feat_mems, mask_mems)
            elif self.cam_compressor is not None:
                global_bev = self.cam_compressor(feat_mems.flatten(1, 2))
            else:
                global_bev = feat_mems.mean(dim=1)
        else:
            global_bev = feat_mems.squeeze(1)

        if has_multi_cameras and S > 1:
            refined_local_bevs = []
            for s in range(S):
                local_bev = feat_mems[:, s]
                if mask_mems is not None:
                    if mask_mems[:, s].abs().sum() == 0:
                        continue
                refined_local = self.gcef(local_bev, global_bev)
                refined_local_bevs.append(refined_local)

            if len(refined_local_bevs) > 0:
                refined_stack = torch.stack(refined_local_bevs, dim=1)
                if mask_mems is not None:
                    valid_masks = [mask_mems[:, s] for s in range(S)
                                   if mask_mems[:, s].abs().sum() > 0]
                    if len(valid_masks) > 0:
                        valid_masks = torch.stack(valid_masks, dim=1)
                        final_global_bev = self._masked_mean_aggregate(refined_stack, valid_masks)
                    else:
                        final_global_bev = refined_stack.mean(dim=1)
                else:
                    final_global_bev = refined_stack.mean(dim=1)
            else:
                final_global_bev = global_bev
        else:
            final_global_bev = self.gcef(global_bev, global_bev)

        return final_global_bev

    def _masked_mean_aggregate(self, feat_mems, mask_mems):
        masked_feat = feat_mems * mask_mems
        sum_feat = masked_feat.sum(dim=1)
        sum_mask = mask_mems.sum(dim=1).clamp(min=1e-6)
        return sum_feat / sum_mask


class GCEFIntegrationMVDet(nn.Module):
    """
    Integrates GCEF into MVDet’s 2D BEV pipeline.
    """

    def __init__(self, feat_dim=128, num_cameras=None, use_camera_compressor=False):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_cameras = num_cameras
        self.gcef = GlobalContextEnhancedFusion(feat_dim=feat_dim)

    def forward(self, feat_mems, mask_mems=None):
        if len(feat_mems.shape) == 5:
            B, S, C, Y, X = feat_mems.shape
            has_multi_cameras = True
        else:
            return feat_mems

        if has_multi_cameras and S > 1:
            if mask_mems is not None:
                masked_feat = feat_mems * mask_mems
                sum_feat = masked_feat.sum(dim=1)
                sum_mask = mask_mems.sum(dim=1).clamp(min=1e-6)
                global_bev = sum_feat / sum_mask
            else:
                global_bev = feat_mems.mean(dim=1)
        else:
            global_bev = feat_mems.squeeze(1)

        if has_multi_cameras and S > 1:
            refined_local_bevs = []
            for s in range(S):
                local_bev = feat_mems[:, s]
                if mask_mems is not None:
                    if mask_mems[:, s].abs().sum() == 0:
                        continue
                refined_local = self.gcef(local_bev, global_bev)
                refined_local_bevs.append(refined_local)

            if len(refined_local_bevs) > 0:
                refined_stack = torch.stack(refined_local_bevs, dim=1)
                if mask_mems is not None:
                    valid_masks = [mask_mems[:, s] for s in range(S)
                                   if mask_mems[:, s].abs().sum() > 0]
                    if len(valid_masks) > 0:
                        valid_masks = torch.stack(valid_masks, dim=1)
                        masked_feat = refined_stack * valid_masks
                        sum_feat = masked_feat.sum(dim=1)
                        sum_mask = valid_masks.sum(dim=1).clamp(min=1e-6)
                        final_global_bev = sum_feat / sum_mask
                    else:
                        final_global_bev = refined_stack.mean(dim=1)
                else:
                    final_global_bev = refined_stack.mean(dim=1)
            else:
                final_global_bev = global_bev
        else:
            final_global_bev = self.gcef(global_bev, global_bev)

        return final_global_bev