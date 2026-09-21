import re
import torch
import torch.nn as nn
import torchvision
import timm
from efficientnet_pytorch import EfficientNet
from timm.utils.model import freeze_batch_norm_2d
from torch.utils.checkpoint import checkpoint as _checkpoint


def _checkpointed(fn, x):
    """Gradient checkpointing wrapper (non-reentrant, RNG-safe).

    Saves/restores the CUDA RNG state around the recompute, so DropPath
    (stochastic depth) inside Swin sees identical masks and the result is
    numerically identical to an un-checkpointed forward/backward.
    """
    return _checkpoint(fn, x, use_reentrant=False)

# ──────────────────────────────────────────────────────────────────
# Pretrained (lightly_train / timm) Swin-T backbone
# ──────────────────────────────────────────────────────────────────
# Same architecture *family* as Encoder_swin_t, but built with timm so
# it can load the checkpoint produced by your lightly_train pretraining
# script (which itself wraps `timm.create_model("swin_tiny_patch4_window7_224", ...)`).
_TIMM_SWIN_NAME = 'swin_tiny_patch4_window7_224'
# Standard Swin-T channel progression (embed_dim=96): stage0/1/2 outputs.
_TIMM_SWIN_OUT_CHANNELS = (96, 192, 384)


def _to_nchw(feat: torch.Tensor, expected_c: int) -> torch.Tensor:
    """timm's Swin returns NCHW in most recent versions but NHWC in
    some configs/older versions. Normalise using the known channel
    count so this class works either way."""
    if feat.dim() == 4 and feat.shape[1] != expected_c and feat.shape[-1] == expected_c:
        return feat.permute(0, 3, 1, 2).contiguous()
    return feat


def _best_prefix_strip(state_dict: dict, target_keys):
    """Find the key-prefix in `state_dict` whose removal overlaps the
    most with `target_keys` (the destination module's own state_dict
    keys).

    Checkpoints from different sources nest the real timm weights
    under different wrapper modules:
      * a lightly_train `.ckpt` Lightning checkpoint
        (e.g. keys like 'model_wrapper._model.layers.0...')
      * your `SwinEmbedder` wrapper's own `_model.` prefix
      * a plain `torch.save(model.state_dict())` export (no prefix)
    Rather than hard-coding one convention, we search for the prefix
    that makes the most keys match.
    """
    target_keys = set(target_keys)
    candidates = {''}
    for k in state_dict.keys():
        parts = k.split('.')
        for i in range(1, len(parts)):
            candidates.add('.'.join(parts[:i]) + '.')
    best_prefix, best_score = '', -1
    for prefix in candidates:
        score = sum(
            1 for k in state_dict
            if k.startswith(prefix) and k[len(prefix):] in target_keys
        )
        if score > best_score:
            best_prefix, best_score = prefix, score
    return best_prefix, best_score


def _layers_dot_to_underscore(k: str) -> str:
    return re.sub(r'(^|\.)layers\.(\d+)\.', r'\1layers_\2.', k)


def _layers_underscore_to_dot(k: str) -> str:
    return re.sub(r'(^|\.)layers_(\d+)\.', r'\1layers.\2.', k)


def load_pretrained_swin_weights(module: nn.Module, path: str,
                                  verbose: bool = True):
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, nn.Module):
        state_dict = obj.state_dict()
    elif isinstance(obj, dict) and 'state_dict' in obj:
        state_dict = obj['state_dict']
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise ValueError(
            f"Unrecognised checkpoint format at {path} ({type(obj)})"
        )
    target_keys = list(module.state_dict().keys())

    # timm's `features_only=True` wrapper renames `layers.<n>.` (vanilla
    # SwinTransformer, used by your SwinEmbedder pretraining script) to
    # `layers_<n>.` (FeatureListNet). Try BOTH spellings and keep
    # whichever overlaps best with `module`'s own keys, so loading works
    # regardless of which flavour produced the checkpoint / which
    # flavour `module` is.
    candidates = {
        'original': state_dict,
        'dot->underscore': {_layers_dot_to_underscore(k): v
                            for k, v in state_dict.items()},
        'underscore->dot': {_layers_underscore_to_dot(k): v
                            for k, v in state_dict.items()},
    }
    best_name, best_prefix, best_score, best_dict = None, '', -1, state_dict
    for name, sd in candidates.items():
        prefix, score = _best_prefix_strip(sd, target_keys)
        if score > best_score:
            best_name, best_prefix, best_score, best_dict = name, prefix, score, sd

    cleaned = {
        k[len(best_prefix):]: v for k, v in best_dict.items()
        if k.startswith(best_prefix)
    }
    missing, unexpected = module.load_state_dict(cleaned, strict=False)
    if verbose:
        print(f"[SwinPretrained] Loading '{path}'")
        print(f"  key spelling used : {best_name}")
        print(f"  matched prefix    : '{best_prefix}' ({best_score} overlapping keys)")
        print(f"  missing keys      : {len(missing)}"
              + (f"  e.g. {missing[:3]}" if missing else ""))
        print(f"  unexpected keys   : {len(unexpected)}"
              + (f"  e.g. {unexpected[:3]}" if unexpected else ""))
        if best_score < 0.5 * len(target_keys):
            print(f"  ⚠ only {best_score}/{len(target_keys)} target keys "
                  f"matched — the backbone is likely running mostly "
                  f"RANDOM weights, NOT your pretrained checkpoint. "
                  f"Inspect state_dict keys manually.")
    return missing, unexpected


class Encoder_swin_t_lightly(nn.Module):
    """Swin-T backbone built with `timm`, loadable from a lightly_train
    checkpoint. Mirrors `Encoder_swin_t`'s FPN decoder EXACTLY (same
    channel counts, same output stride = input/4) so it is a drop-in
    replacement — only the backbone implementation + weight source
    differ.
    """
    def __init__(self, C, pretrained_path=None, freeze_backbone=False,
                 grad_checkpoint=False):
        super().__init__()
        self.C = C
        # Gradient checkpointing: recompute the backbone during backward
        # instead of retaining every activation of a 720x1280 fp32 Swin
        # forward. ~3-4x lower backbone activation memory for ~1.3x step
        # time. Numerics identical (checkpoint saves/restores RNG state,
        # so DropPath masks are the same).
        self.grad_checkpoint = bool(grad_checkpoint)
        if self.grad_checkpoint:
            print("[MEM] image-backbone gradient checkpointing ENABLED "
                  "(expect ~30% slower steps; several GiB lower peak)")
        create_kwargs = dict(
            pretrained=(pretrained_path is None),  # fall back to timm's
            num_classes=0,                         # own ImageNet weights
            features_only=True,                    # if no custom ckpt
            out_indices=(0, 1, 2),
        )
        try:
            self.backbone = timm.create_model(
                _TIMM_SWIN_NAME,
                img_size=None,
                strict_img_size=False,
                always_partition=True,
                **create_kwargs,
            )
        except TypeError:
            try:
                self.backbone = timm.create_model(
                    _TIMM_SWIN_NAME,
                    img_size=None,
                    strict_img_size=False,
                    always_partition=True,
                    **create_kwargs,
                )
            except TypeError:
                try:
                    self.backbone = timm.create_model(
                        _TIMM_SWIN_NAME, strict_img_size=False, **create_kwargs
                    )
                except TypeError:
                    self.backbone = timm.create_model(_TIMM_SWIN_NAME, **create_kwargs)
                    print("[SwinPretrained] WARNING: your timm version does not "
                          "support strict_img_size=False for Swin — inputs MUST "
                          "be exactly 224x224 or you will hit the PatchEmbed "
                          "assertion. Run `pip install -U timm`.")
        if pretrained_path:
            load_pretrained_swin_weights(self.backbone, pretrained_path)
        freeze_bn(self.backbone)  # no-op for LayerNorm-based Swin; kept
        if freeze_backbone:       # for parity with Encoder_swin_t
            for p in self.backbone.parameters():
                p.requires_grad = False
        c0, c1, c2 = _TIMM_SWIN_OUT_CHANNELS
        self.upsampling_layer1 = UpsamplingConcat(c2 + c1, c2)
        self.upsampling_layer2 = UpsamplingConcat(c2 + c0, c2)
        self.depth_layer = nn.Conv2d(c2, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        orig_dtype = x.dtype
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                feats = _checkpointed(self.backbone, x)
            else:
                feats = self.backbone(x)

            c0, c1, c2 = _TIMM_SWIN_OUT_CHANNELS
            x0 = _to_nchw(feats[0], c0)
            x1 = _to_nchw(feats[1], c1)
            x2 = _to_nchw(feats[2], c2)
            x = self.upsampling_layer1(x2, x1)
            x = self.upsampling_layer2(x, x0)
            x = self.depth_layer(x)
        return x.to(orig_dtype)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum


def freeze_bn(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            freeze_bn(module)

        if isinstance(module, torch.nn.BatchNorm2d):
            setattr(model, n, freeze_batch_norm_2d(module))


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class Encoder_swin_t(nn.Module):
    def __init__(self, C, grad_checkpoint=False):
        super().__init__()
        self.C = C
        # See Encoder_swin_t_lightly: activation-memory/speed trade,
        # numerics identical (RNG state is saved/restored on recompute).
        self.grad_checkpoint = bool(grad_checkpoint)
        if self.grad_checkpoint:
            print("[MEM] image-backbone gradient checkpointing ENABLED "
                  "(expect ~30% slower steps; several GiB lower peak)")
        swin_t = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.DEFAULT)
        freeze_bn(swin_t)
        self.layer0 = swin_t.features[0]
        self.layer1 = swin_t.features[1:3]
        self.layer2 = swin_t.features[3:5]
        # self.layer3 = swin_t.features[5:7]

        self.upsampling_layer1 = UpsamplingConcat(384 + 192, 384)
        self.upsampling_layer2 = UpsamplingConcat(384 + 96, 384)
        self.depth_layer = nn.Conv2d(384, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        orig_dtype = x.dtype
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                x0 = _checkpointed(self.layer0, x)
                x1 = _checkpointed(self.layer1, x0)
                x2 = _checkpointed(self.layer2, x1)
            else:
                x0 = self.layer0(x)
                x1 = self.layer1(x0)
                x2 = self.layer2(x1)

            x = self.upsampling_layer1(
                x2.permute(0, 3, 1, 2), x1.permute(0, 3, 1, 2))
            x = self.upsampling_layer2(x, x0.permute(0, 3, 1, 2))
            x = self.depth_layer(x)
        return x.to(orig_dtype)


class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        freeze_bn(resnet)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.upsampling_layer1 = UpsamplingConcat(1024 + 512, 512)
        self.upsampling_layer2 = UpsamplingConcat(512 + 256, 512)
        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x


class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        freeze_bn(resnet)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.upsampling_layer1 = UpsamplingConcat(1024 + 512, 512)
        self.upsampling_layer2 = UpsamplingConcat(512 + 256,  512)
        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x


class Encoder_res34(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
        freeze_bn(resnet)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.upsampling_layer1 = UpsamplingConcat(256 + 128, 256)
        self.upsampling_layer2 = UpsamplingConcat(256 + 64, 256)
        self.depth_layer = nn.Conv2d(256, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x

class Encoder_res18(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        freeze_bn(resnet)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.upsampling_layer1 = UpsamplingConcat(256 + 128, 256)
        self.upsampling_layer2 = UpsamplingConcat(256 + 64, 256)
        self.depth_layer = nn.Conv2d(256, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x


class Encoder_eff(nn.Module):
    def __init__(self, C, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8
        self.version = version

        if self.version == 'b0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.version == 'b4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        # print('input_1', input_1.shape)
        # print('input_2', input_2.shape)
        x = self.upsampling_layer(input_1, input_2)
        # print('x', x.shape)
        return x

    def forward(self, x):
        x = self.get_features(x)  # get feature vector
        x = self.depth_layer(x)  # feature and depth head
        return x