import torch
import torch.nn as nn

import utils.geom
import utils.vox
import utils.basic

from kornia.geometry.transform.imgwarp import warp_perspective
from models.encoder import Encoder_res101, Encoder_res50, Encoder_res18, Encoder_eff, Encoder_swin_t, Encoder_res34
from models.decoder import Decoder
from models.gcef import GCEFIntegrationMVDet


class MVDet(nn.Module):
    def __init__(self, Y, Z, X,
                 rand_flip=False,
                 num_cameras=None,
                 num_classes=None,
                 latent_dim=256,
                 feat2d_dim=128,
                 encoder_type='res18',
                 use_gcef=True,
                 device=torch.device('cuda')):
        super().__init__()
        assert (encoder_type in ['res101', 'res50', 'res18', 'res34', 'effb0', 'effb4', 'swin_t'])

        self.Y, self.Z, self.X = Y, Z, X
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.num_cameras = num_cameras
        self.use_gcef = use_gcef
        self.mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
        self.std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)

        # Encoder
        self.feat2d_dim = feat2d_dim
        if encoder_type == 'res101':
            self.encoder = Encoder_res101(self.feat2d_dim)
        elif encoder_type == 'res50':
            self.encoder = Encoder_res50(self.feat2d_dim)
        elif encoder_type == 'effb0':
            self.encoder = Encoder_eff(self.feat2d_dim, version='b0')
        elif encoder_type == 'res18':
            self.encoder = Encoder_res18(self.feat2d_dim)
        elif encoder_type == 'res34':
            self.encoder = Encoder_res34(self.feat2d_dim)
        elif encoder_type == 'swin_t':
            self.encoder = Encoder_swin_t(self.feat2d_dim)
        else:
            self.encoder = Encoder_eff(self.feat2d_dim, version='b4')

        # Camera aggregation
        # GCEF integration (2D version for MVDet)
        if self.use_gcef:
            from models.gcef import GCEFIntegrationMVDet
            self.gcef_module = GCEFIntegrationMVDet(
                feat_dim=feat2d_dim,  # 128
                num_cameras=num_cameras
            )
            # Compressor to go from feat2d_dim to latent_dim
            self.feat_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim, latent_dim, kernel_size=3, padding=1, stride=1),
                nn.InstanceNorm2d(latent_dim),
                nn.ReLU(),
                nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            )
        else:
            # Original aggregation
            if self.num_cameras is not None:
                self.cam_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim * self.num_cameras, latent_dim,
                              kernel_size=3, padding=1, stride=1),
                    nn.InstanceNorm2d(latent_dim),
                    nn.ReLU(),
                    nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
                )
            self.feat_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim, latent_dim, kernel_size=3, padding=1, stride=1),
                nn.InstanceNorm2d(latent_dim),
                nn.ReLU(),
                nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            )

        # Temporal BEV fusion
        self.temporal_bev = nn.Sequential(
            nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(latent_dim), nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
        )

        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=num_classes,
            feat2d=self.feat2d_dim,
        )

        # Weights
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.tracking_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.size_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.rot_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_img_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, rgb_cams, pix_T_cams, cams_T_global, vox_util, ref_T_global, prev_bev=None, Y=None, Z=None,
                X=None):
        """
        Forward pass with optional dynamic resolution
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width

        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        vox_util: vox util object
        ref_T_global: (B,4,4)
        Y, Z, X: Optional resolution override (for multi-dataset training)
        """
        # Use provided resolution or default
        if Y is None or Z is None or X is None:
            Y, Z, X = self.Y, self.Z, self.X

        B, S, C, H, W = rgb_cams.shape

        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_cams_ = __p(rgb_cams)  # B*S,3,H,W
        pix_T_cams_ = __p(pix_T_cams)  # B*S,4,4
        cams_T_global_ = __p(cams_T_global)  # B*S,4,4

        global_T_cams_ = torch.inverse(cams_T_global_)  # B*S,4,4
        ref_T_cams_ = torch.matmul(ref_T_global.repeat(S, 1, 1), global_T_cams_)  # B*S,4,4
        cams_T_ref_ = torch.inverse(ref_T_cams_)  # B*S,4,4

        # rgb encoder
        device = rgb_cams_.device
        rgb_cams_ = (rgb_cams_ - self.mean.to(device)) / self.std.to(device)  # B*S,3,H,W
        feat_cams_ = self.encoder(rgb_cams_)  # B*S,feat2d_dim,H/8,W/8
        _, C, Hf, Wf = feat_cams_.shape

        sy = Hf / float(H)
        sx = Wf / float(W)
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)  # B*S,4,4
        featpix_T_ref_ = torch.matmul(featpix_T_cams_[:, :3, :3], cams_T_ref_[:, :3, [0, 1, 3]])  # B*S,3,3

        # Use dynamic Y, X for projection matrix
        ref_T_mem = vox_util.get_ref_T_mem(B, Y, Z, X)  # B,4,4
        ref_T_mem = ref_T_mem[0, [0, 1, 3]][:, [0, 1, 3]]  # 3,3
        featpix_T_mem_ = torch.matmul(featpix_T_ref_, ref_T_mem)  # B*S,3,3
        mem_T_featpix = torch.inverse(featpix_T_mem_)  # B*S,3,3
        proj_mats = mem_T_featpix  # B*S,3,3

        # Warp to BEV with dynamic resolution (Y, X)
        # Warp to BEV: B*S, feat2d_dim, Y, X
        feat_mems_ = warp_perspective(feat_cams_, proj_mats, (Y, X), align_corners=False)
        feat_mems = __u(feat_mems_)  # B, S, feat2d_dim, Y, X

        # GCEF INTEGRATION (2D)
        if self.use_gcef:
            # Create masks
            mask_mems = (torch.abs(feat_mems) > 0).float()

            # Apply 2D GCEF
            feat_mem = self.gcef_module(feat_mems, mask_mems)  # B, feat2d_dim, Y, X

            # Compress to latent_dim
            feat_mem = self.feat_compressor(feat_mem)  # B, latent_dim, Y, X

        else:
            # Original aggregation
            if self.num_cameras is None:
                mask_mems = (torch.abs(feat_mems) > 0).float()
                feat_mem = utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1)
                feat_mem = self.feat_compressor(feat_mem)
            else:
                feat_mem = self.cam_compressor(feat_mems.flatten(1, 2))

        # Temporal BEV fusion with resolution matching
        if prev_bev is None:
            prev_bev = feat_mem
        else:
            # Handle resolution mismatch in temporal cache
            if prev_bev.shape[-2:] != feat_mem.shape[-2:]:
                # Previous BEV is from different dataset with different resolution
                # Resize to current resolution
                prev_bev = F.interpolate(prev_bev, size=(Y, X), mode='bilinear', align_corners=False)

        bev_features = torch.cat([feat_mem, prev_bev], dim=1)  # B, latent_dim*2, Y, X
        bev_features = self.temporal_bev(bev_features)  # B, latent_dim, Y, X

        # Decoder handles any resolution via 1x1 convs
        out_dict = self.decoder(
            bev_features,
            feat_cams_,
            (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None
        )

        return out_dict