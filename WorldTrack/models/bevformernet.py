# WorldTrack/models/bevformernet.py
import torch
import torch.nn as nn
import utils.geom
import utils.vox
import utils.basic
from utils.precision import fp32_inverse
from models.encoder import (Encoder_res101, Encoder_res50, Encoder_eff,
                             Encoder_swin_t, Encoder_res18)
from models.decoder import Decoder
from models.ops.ms_deform_attn import MSDeformAttn, MSDeformAttn3D


class VanillaSelfAttention(nn.Module):
    def __init__(self, dim=128, dropout=0.5):
        super(VanillaSelfAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn(
            d_model=dim, n_levels=1, n_heads=4, n_points=8
        )
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, Y, X, query_pos=None):
        inp_residual = query.clone()
        if query_pos is not None:
            query = query + query_pos
        B, N, C = query.shape
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, Y - 0.5, Y, dtype=torch.float,
                           device=query.device),
            torch.linspace(0.5, X - 0.5, X, dtype=torch.float,
                           device=query.device),
            indexing='ij'
        )
        ref_y = ref_y.reshape(-1)[None] / Y
        ref_x = ref_x.reshape(-1)[None] / X
        reference_points = torch.stack((ref_y, ref_x), -1)
        reference_points = reference_points.repeat(B, 1, 1).unsqueeze(2)
        input_spatial_shapes = query.new_zeros([1, 2]).long()
        input_spatial_shapes[0, 0] = Y
        input_spatial_shapes[0, 1] = X
        input_level_start_index = query.new_zeros([1, ]).long()

        # ms_deform_attn CUDA kernel does not support bf16/fp16.
        # Disable autocast and cast all float tensors to fp32 explicitly,
        # then cast the output back to the original dtype.
        orig_dtype = query.dtype
        with torch.amp.autocast('cuda', enabled=False):
            queries = self.deformable_attention(
                query=query.float(),
                reference_points=reference_points.float(),
                input_flatten=query.clone().float(),
                input_spatial_shapes=input_spatial_shapes,
                input_level_start_index=input_level_start_index,
            )
        queries = queries.to(orig_dtype)

        queries = self.output_proj(queries)
        return self.dropout(queries) + inp_residual


class SpatialCrossAttention(nn.Module):
    def __init__(self, dim=128, dropout=0.5):
        super(SpatialCrossAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn3D(
            embed_dims=dim, num_heads=4, num_levels=1, num_points=8
        )
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value, query_pos=None,
                reference_points_cam=None, spatial_shapes=None,
                bev_mask=None):
        inp_residual = query
        slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos
        B, N, C = query.shape
        S, M, _, _ = key.shape
        D = reference_points_cam.size(3)
        max_len = bev_mask.sum(dim=-1).gt(0).sum(-1).max()
        queries_rebatch = query.new_zeros([B, S, max_len, self.dim])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [B, S, max_len, D, 2]
        )
        for j in range(B):
            for i, reference_points_per_img in enumerate(
                    reference_points_cam):
                index_query_per_img = (
                    bev_mask[i, j].sum(-1).nonzero().squeeze(-1)
                )
                queries_rebatch[j, i, :len(index_query_per_img)] = (
                    query[j, index_query_per_img]
                )
                reference_points_rebatch[
                j, i, :len(index_query_per_img)
                ] = reference_points_per_img[j, index_query_per_img]

        key = key.permute(2, 0, 1, 3).reshape(B * S, M, C)
        value = value.permute(2, 0, 1, 3).reshape(B * S, M, C)
        level_start_index = query.new_zeros([1, ]).long()
        reference_points_rebatch = reference_points_rebatch.view(
            B * S, max_len, D, 2
        )

        # ms_deform_attn CUDA kernel does not support bf16/fp16.
        # Disable autocast and cast all float tensors to fp32 explicitly,
        # then cast the output back to the original dtype.
        orig_dtype = queries_rebatch.dtype
        with torch.amp.autocast('cuda', enabled=False):
            queries = self.deformable_attention(
                query=queries_rebatch.view(B * S, max_len, self.dim).float(),
                key=key.float(),
                value=value.float(),
                reference_points=reference_points_rebatch.float(),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            ).view(B, S, max_len, self.dim)
        queries = queries.to(orig_dtype)

        for j in range(B):
            for i in range(S):
                index_query_per_img = (
                    bev_mask[i, j].sum(-1).nonzero().squeeze(-1)
                )
                slots[j, index_query_per_img] += (
                    queries[j, i, :len(index_query_per_img)]
                )
        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)
        return self.dropout(slots) + inp_residual


class Bevformernet(nn.Module):
    def __init__(self, Y, Z, X,
                 rand_flip=False,
                 latent_dim=128,
                 feat2d_dim=128,
                 num_classes=None,
                 z_sign=1,
                 encoder_type='swin_t',
                 use_image_aux=False):
        super(Bevformernet, self).__init__()
        assert encoder_type in [
            'res101', 'res50', 'res18', 'effb0', 'effb4', 'swin_t'
        ]
        self.Y, self.Z, self.X = Y, Z, X
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.use_radar = False
        self.use_lidar = False
        self.register_buffer(
            'mean',
            torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            'std',
            torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1),
        )
        self.z_sign = z_sign

        # Encoder
        self.feat2d_dim = feat2d_dim
        if encoder_type == 'res101':
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == 'res50':
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == 'res18':
            self.encoder = Encoder_res18(feat2d_dim)
        elif encoder_type == 'effb0':
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        elif encoder_type == 'swin_t':
            self.encoder = Encoder_swin_t(feat2d_dim)
        else:
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        # BEVFormer self & cross attention layers
        self.bev_keys = nn.Linear(feat2d_dim, latent_dim)
        self.bev_queries = nn.Parameter(
            0.1 * torch.randn(latent_dim, Y, X)
        )
        self.bev_queries_pos = nn.Parameter(
            0.1 * torch.randn(latent_dim, Y, X)
        )
        num_layers = 6
        self.num_layers = num_layers
        self.self_attn_layers = nn.ModuleList([
            VanillaSelfAttention(dim=latent_dim)
            for _ in range(num_layers)
        ])
        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            SpatialCrossAttention(dim=latent_dim)
            for _ in range(num_layers)
        ])
        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])
        ffn_dim = 512
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, ffn_dim), nn.ReLU(),
                nn.Linear(ffn_dim, latent_dim),
            )
            for _ in range(num_layers)
        ])
        self.norm3_layers = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])

        self.bev_temporal = nn.Sequential(
            nn.Conv2d(latent_dim * 2, latent_dim,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(latent_dim), nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
        )

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=num_classes,
            feat2d=feat2d_dim,
            use_image_aux=use_image_aux,
        )

        # Weights
        self.center_weight = nn.Parameter(
            torch.tensor(0.0), requires_grad=True
        )
        self.offset_weight = nn.Parameter(
            torch.tensor(0.0), requires_grad=True
        )
        self.tracking_weight = nn.Parameter(
            torch.tensor(0.0), requires_grad=True
        )

    def forward(self, rgb_cams, pix_T_cams, cams_T_global, vox_util,
                ref_T_global, prev_bev=None):
        B, S, C, H, W = rgb_cams.shape
        B0 = B * S
        assert C == 3

        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)

        rgb_cams_ = __p(rgb_cams)
        pix_T_cams_ = __p(pix_T_cams)
        cams_T_global_ = __p(cams_T_global)
        global_T_cams_ = fp32_inverse(cams_T_global_)
        ref_T_cams = torch.matmul(
            ref_T_global.repeat(S, 1, 1), global_T_cams_
        )
        cams_T_ref_ = fp32_inverse(ref_T_cams)

        device = rgb_cams_.device
        rgb_cams_ = (
                            rgb_cams_
                            - self.mean.to(device=device, dtype=rgb_cams_.dtype)
                    ) / self.std.to(device=device, dtype=rgb_cams_.dtype)
        feat_cams_ = self.encoder(rgb_cams_)
        _, Cf, Hf, Wf = feat_cams_.shape
        feat_cams = __u(feat_cams_)

        Y, Z, X = self.Y, self.Z, self.X

        xyz_mem_ = utils.basic.gridcloud3d(
            B0, Y, Z, X, norm=False, device=rgb_cams.device
        )
        xyz_ref_ = vox_util.Mem2Ref(xyz_mem_, Y, Z, X, assert_cube=False)
        xyz_cams_ = utils.geom.apply_4x4(cams_T_ref_, xyz_ref_)
        xy_cams_ = utils.geom.camera2pixels(xyz_cams_, pix_T_cams_)

        reference_points_cam = (
            xy_cams_.reshape(B, S, Y, Z, X, 2)
            .permute(1, 0, 2, 4, 3, 5)
            .reshape(S, B, Y * X, Z, 2)
        )
        reference_points_cam[..., 0:1] = (
            reference_points_cam[..., 0:1] / float(W)
        )
        reference_points_cam[..., 1:2] = (
            reference_points_cam[..., 1:2] / float(H)
        )
        cam_x = (
            xyz_cams_[..., 2]
            .reshape(B, S, Y, Z, X, 1)
            .permute(1, 0, 2, 4, 3, 5)
            .reshape(S, B, Y * X, Z, 1)
        )
        bev_mask = (
            (reference_points_cam[..., 1:2] >= 0.0)
            & (reference_points_cam[..., 1:2] <= 1.0)
            & (reference_points_cam[..., 0:1] <= 1.0)
            & (reference_points_cam[..., 0:1] >= 0.0)
            & (self.z_sign * cam_x >= 0.0)
        ).squeeze(-1)

        bev_queries = (
            self.bev_queries.clone()
            .unsqueeze(0)
            .repeat(B, 1, 1, 1)
            .reshape(B, self.latent_dim, -1)
            .permute(0, 2, 1)
        )
        bev_queries_pos = (
            self.bev_queries_pos.clone()
            .unsqueeze(0)
            .repeat(B, 1, 1, 1)
            .reshape(B, self.latent_dim, -1)
            .permute(0, 2, 1)
        )

        bev_keys = (
            feat_cams.reshape(B, S, Cf, Hf * Wf).permute(1, 3, 0, 2)
        )
        bev_keys = self.bev_keys(bev_keys)
        spatial_shapes = bev_queries.new_zeros([1, 2]).long()
        spatial_shapes[0, 0] = Hf
        spatial_shapes[0, 1] = Wf

        for i in range(self.num_layers):
            bev_queries = self.self_attn_layers[i](
                bev_queries, self.Y, self.X, bev_queries_pos
            )
            bev_queries = self.norm1_layers[i](bev_queries)
            bev_queries = self.cross_attn_layers[i](
                bev_queries, bev_keys, bev_keys,
                query_pos=bev_queries_pos,
                reference_points_cam=reference_points_cam,
                spatial_shapes=spatial_shapes,
                bev_mask=bev_mask,
            )
            bev_queries = self.norm2_layers[i](bev_queries)
            bev_queries = (
                bev_queries + self.ffn_layers[i](bev_queries)
            )
            bev_queries = self.norm3_layers[i](bev_queries)

        feat_bev = bev_queries.permute(0, 2, 1).reshape(
            B, self.latent_dim, self.Y, self.X
        )

        if prev_bev is None:
            prev_bev = feat_bev
        feat_bev = torch.cat([feat_bev, prev_bev], dim=1)
        feat_bev = self.bev_temporal(feat_bev)

        out_dict = self.decoder(feat_bev, feat_cams_)
        return out_dict