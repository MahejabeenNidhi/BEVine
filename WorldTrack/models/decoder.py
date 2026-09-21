# WorldTrack/models/decoder.py
import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from models.encoder import freeze_bn, UpsamplingConcat
from utils.basic import SIZE_PRIOR_M


class Attr3DQueryHead(nn.Module):
    """Per-object 3D attribute head (yaw / size / posture).

    Attributes are predicted from features GATHERED at each object location
    instead of from a dense per-pixel readout:

      1. bilinear-sample the dedicated attribute neck at the (sub-pixel)
         object centre -> base query vector;
      2. predict S sampling offsets + weights from that query and gather
         context around the object (single-scale deformable attention
         expressed with grid_sample -- no custom CUDA op, bf16-safe);
      3. fuse query + context through a small MLP and decode yaw/size/
         posture with one linear layer each.

    The SAME query() is used at training time (GT centres) and at test time
    (decoded peaks), so there is no train/test sampling mismatch.
    """

    def __init__(self, feat_dim, head_dim=128, n_points=8,
                 max_offset_cells=20.0, pos_dim=32):
        super().__init__()
        self.n_points = int(n_points)
        # 20 cells = 2 m at 10 cm/cell: ~a full cow length, reachable from
        # the centre. tanh keeps the very first (untrained) steps local.
        self.max_offset_cells = float(max_offset_cells)

        self.pos_proj = nn.Linear(2, pos_dim)
        self.query_proj = nn.Linear(feat_dim + pos_dim, head_dim)
        self.offset_proj = nn.Linear(head_dim, self.n_points * 2)
        self.weight_proj = nn.Linear(head_dim, self.n_points)
        self.ctx_proj = nn.Linear(feat_dim, head_dim)
        self.fuse = nn.Linear(2 * head_dim, head_dim)
        self.norm = nn.LayerNorm(head_dim)

        self.yaw_head = nn.Linear(head_dim, 2)       # (sin 2t, cos 2t)
        self.size_head = nn.Linear(head_dim, 3)      # (l, w, h) metres
        self.posture_head = nn.Linear(head_dim, 1)   # standing logit

        # Start from the same priors the dense heads used, so early-loss
        # magnitudes are unchanged. Offsets start at zero -> context ==
        # centre feature at init (graceful degradation to a point query).
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)
        self.yaw_head.bias.data = torch.tensor([0.0, 1.0])
        self.yaw_head.weight.data.mul_(0.1)
        self.size_head.bias.data = torch.tensor(
            SIZE_PRIOR_M, dtype=torch.float32)
        self.size_head.weight.data.mul_(0.1)
        nn.init.zeros_(self.posture_head.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _bilinear(feat, xy):
        """feat (B,C,Y,X), xy (B,N,2) float (x,y) cell coords -> (B,N,C).

        'border' padding: an object near the arena edge (or a learned
        offset pointing outside) samples the clamped edge feature instead
        of zeros.
        """
        B, C, Y, X = feat.shape
        gx = (xy[..., 0] + 0.5) / float(X) * 2.0 - 1.0
        gy = (xy[..., 1] + 0.5) / float(Y) * 2.0 - 1.0
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(2)      # (B,N,1,2)
        out = F.grid_sample(feat, grid, mode='bilinear',
                            padding_mode='border',
                            align_corners=False)               # (B,C,N,1)
        return out.squeeze(-1).permute(0, 2, 1)                # (B,N,C)

    # ------------------------------------------------------------------
    def query(self, attr_feat, xy):
        """attr_feat (B,C,Y,X), xy (B,N,2) float BEV memory coords.

        Returns raw per-object predictions:
            yaw     (B,N,2) raw (sin 2t, cos 2t) -- NOT normalised
            size    (B,N,3) metres
            posture (B,N,1) logit

        Runs in fp32: bf16 would quantise the sub-cell sample locations.
        """
        B, C, Y, X = attr_feat.shape
        N = xy.shape[1]
        feat = attr_feat.float()
        xy = xy.float()

        base = self._bilinear(feat, xy)                        # (B,N,C)
        pos = self.pos_proj(xy / xy.new_tensor([float(X), float(Y)]))
        q = self.query_proj(torch.cat((base, pos), dim=-1))    # (B,N,D)

        off = self.max_offset_cells * torch.tanh(
            self.offset_proj(q)).view(B, N, self.n_points, 2)
        pts = xy.unsqueeze(2) + off                            # (B,N,S,2)
        ctx = self._bilinear(feat, pts.reshape(B, N * self.n_points, 2))
        ctx = ctx.view(B, N, self.n_points, C)
        w = torch.softmax(self.weight_proj(q), dim=-1)         # (B,N,S)
        c = (ctx * w.unsqueeze(-1)).sum(dim=2)                 # (B,N,C)

        h = q + self.fuse(torch.cat((q, self.ctx_proj(c)), dim=-1))
        h = self.norm(h)

        return {
            'yaw': self.yaw_head(h),          # (B,N,2)
            'size': self.size_head(h),        # (B,N,3) metres
            'posture': self.posture_head(h),  # (B,N,1)
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def query_extra(self, attr_feat, xy, size_scale=100.0):
        """Decode-compatible attribute dict: the same keys/units the dense
        gather in utils.decode.decoder used to produce."""
        raw = self.query(attr_feat, xy)
        yaw_sc = raw['yaw']
        yaw_sc = yaw_sc / yaw_sc.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        post_p = torch.sigmoid(raw['posture'].squeeze(-1))     # (B,N)
        return {
            'yaw_sincos': yaw_sc,
            'yaw_angle': 0.5 * torch.atan2(yaw_sc[..., 0], yaw_sc[..., 1]),
            'dimensions': raw['size'].clamp(min=0.0) * float(size_scale),
            'posture_prob': post_p,
            'posture_class': (post_p > 0.5).long(),
        }


class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, feat2d=128,
                 use_image_aux=False, bev3d_grad_scale=1.0,
                 attr3d_neck_dim=128, attr3d_head_dim=128,
                 attr3d_points=8, attr3d_max_offset_cells=20.0,
                 attr3d_detach_trunk=True):
        super().__init__()
        self.bev3d_grad_scale = bev3d_grad_scale
        self.attr3d_detach_trunk = bool(attr3d_detach_trunk)

        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        freeze_bn(backbone)
        self.first_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.feat2d = feat2d
        self.head_conv = 128

        self.up3_skip = UpsamplingConcat(256 + 128, 256)
        self.up2_skip = UpsamplingConcat(256 + 64, 256)
        self.up1_skip = UpsamplingConcat(256 + in_channels, in_channels)

        # BEV heads
        # Dense heads: DETECTION ONLY. yaw/size/posture moved to the
        # per-object query head below.
        self.bev_heads = nn.ModuleDict()
        bev_head_config = {
            'center': n_classes,
            'offset': 4,
        }
        for name, out_channels in bev_head_config.items():
            self.bev_heads[name] = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(self.head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv, out_channels,
                          kernel_size=1, padding=0),
            )
            if name == 'center':
                self.bev_heads[name][-1].bias.data.fill_(-2.19)

        # dedicated 3D attribute neck + per-object query head
        # The neck is the ONLY consumer of attribute gradients when the
        # trunk input is detached (see forward): full-strength learning
        # for attributes, zero interference with centre/offset features.
        self.attr_neck = nn.Sequential(
            nn.Conv2d(in_channels, attr3d_neck_dim,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(attr3d_neck_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(attr3d_neck_dim, attr3d_neck_dim,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(attr3d_neck_dim),
            nn.ReLU(inplace=True),
        )
        self.attr3d_head = Attr3DQueryHead(
            feat_dim=attr3d_neck_dim,
            head_dim=attr3d_head_dim,
            n_points=attr3d_points,
            max_offset_cells=attr3d_max_offset_cells,
        )

        # Image heads
        self.img_heads = nn.ModuleDict()
        self.img_heads_config = {'center': n_classes}
        if use_image_aux:
            self.img_heads_config['offset'] = 2
            self.img_heads_config['size'] = 2

        for name, out_channels in self.img_heads_config.items():
            self.img_heads[name] = nn.Sequential(
                nn.Conv2d(self.feat2d, self.feat2d,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(self.feat2d),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.feat2d, out_channels,
                          kernel_size=1, padding=0),
            )
            if name == 'center':
                self.img_heads[name][-1].bias.data.fill_(-2.19)

    # ------------------------------------------------------------------
    def forward(self, x, feat_cams, bev_flip_indices=None):
        b, c, h, w = x.shape
        x_raw = x

        # pad input
        m = 16
        ph = math.ceil(h / m) * m - h
        pw = math.ceil(w / m) * m - w
        pt, pb = ph // 2, ph - (ph // 2)
        pl, pr = pw // 2, pw - (pw // 2)
        x = torch.nn.functional.pad(x, [pl, pr, pt, pb])

        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x
        x = self.layer3(x)

        x = self.up3_skip(x, skip_x['3'])
        x = self.up2_skip(x, skip_x['2'])
        x = self.up1_skip(x, skip_x['1'])

        x = x[..., pt:pt + h, pl:pl + w]

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2])
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        # BEV outputs
        out_bev = {'bev_raw': x_raw, 'bev_feat': x}
        for name, head in self.bev_heads.items():
            out_bev[f'instance_{name}'] = head(x)

        # Dedicated attribute neck. With attr3d_detach_trunk the trunk is
        # detached: the attribute loss trains the NECK + QUERY HEAD at
        # full strength and can no longer perturb centre/offset features
        # (replaces the old grad_scale attenuation outright).
        neck_in = x.detach() if self.attr3d_detach_trunk else x
        out_bev['attr_feat'] = self.attr_neck(neck_in)

        # Image outputs
        out_img = {'img_raw_feat': feat_cams}
        for name, head in self.img_heads.items():
            out_img[f'img_{name}'] = head(feat_cams)

        return {**out_bev, **out_img}
