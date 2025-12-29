import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    Resolution-adaptive BEV decoder for multi-dataset training.

    Handles variable Y, X dimensions while maintaining spatial consistency.
    Uses 1×1 convolutions for prediction heads (resolution-independent).
    """

    def __init__(
            self,
            in_channels=256,
            n_classes=1,
            feat2d=128,
    ):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes

        # Shared BEV feature processing
        # All conv layers use padding to maintain spatial dimensions
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Detection heads (resolution-independent via 1×1 convs)
        self.center_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1),  # 1×1 conv = resolution-independent
        )

        # Offset: (x, y, prev_x, prev_y) = 4 channels
        self.offset_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=1),
        )

        # Size: (width, height) = 2 channels
        self.size_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

        # Rotation: (sin, cos) × n_bins
        self.rot_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=1),  # 8 bins × 2 (sin, cos)
        )

        # Image center prediction head (for multi-view consistency)
        self.img_center_head = nn.Sequential(
            nn.Conv2d(feat2d, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

    def forward(self, bev_features, img_features, flip_indices=None):
        """
        Forward pass with resolution-adaptive processing.

        Args:
            bev_features: (B, in_channels, Y, X) - variable Y, X dimensions
            img_features: (B*S, feat2d, H/8, W/8) - image features for auxiliary loss
            flip_indices: Optional flip augmentation indices

        Returns:
            Dictionary with predictions at the current resolution
        """
        B, C, Y, X = bev_features.shape

        # BEV feature processing (handles any Y, X size)
        x = self.conv1(bev_features)  # B, 256, Y, X
        x = self.conv2(x)  # B, 256, Y, X
        x = self.conv3(x)  # B, 128, Y, X

        # Detection heads (all use 1×1 convs, so resolution-independent)
        center = self.center_head(x)  # B, n_classes, Y, X
        offset = self.offset_head(x)  # B, 4, Y, X
        size = self.size_head(x)  # B, 2, Y, X
        rot = self.rot_head(x)  # B, 16, Y, X

        # Handle augmentation flips if needed
        if flip_indices is not None:
            flip1_index, flip2_index = flip_indices
            center[flip1_index] = torch.flip(center[flip1_index], [-1])
            center[flip2_index] = torch.flip(center[flip2_index], [-3])
            offset[flip1_index] = torch.flip(offset[flip1_index], [-1])
            offset[flip2_index] = torch.flip(offset[flip2_index], [-3])

        # Image center prediction (for multi-view consistency loss)
        img_center = self.img_center_head(img_features)  # B*S, n_classes, H/8, W/8

        return {
            'bev_raw': bev_features,  # B, in_channels, Y, X (for temporal cache)
            'instance_center': center,  # B, n_classes, Y, X
            'instance_offset': offset,  # B, 4, Y, X
            'instance_size': size,  # B, 2, Y, X
            'instance_rot': rot,  # B, 16, Y, X
            'img_center': img_center,  # B*S, n_classes, H/8, W/8
        }


class AspectRatioNormalizedDecoder(Decoder):
    """
    Decoder with aspect-ratio normalization for extreme resolution differences.

    Use this ONLY if you observe:
    1. Significant performance degradation on one dataset vs another
    2. Very different aspect ratios (e.g., 2:1 vs 1:2)
    3. Visual artifacts in predictions

    For TrackTacular (200×120, ratio 1.67) vs MMCows (117×192, ratio 0.61),
    this may help if standard Decoder shows issues.

    How it works:
    1. Normalize input to canonical aspect ratio via adaptive pooling
    2. Process with CNN (learns on consistent aspect ratio)
    3. Upsample back to original resolution
    """

    def __init__(
            self,
            in_channels=256,
            n_classes=1,
            feat2d=128,
            canonical_size=(128, 128),  # Square intermediate resolution
    ):
        super().__init__(in_channels, n_classes, feat2d)
        self.canonical_size = canonical_size

        print(f"\n{'=' * 60}")
        print("WARNING: Using AspectRatioNormalizedDecoder")
        print(f"Canonical size: {canonical_size}")
        print("This adds computational overhead. Use only if standard Decoder fails.")
        print(f"{'=' * 60}\n")

    def forward(self, bev_features, img_features, flip_indices=None):
        """
        Forward with aspect ratio normalization.

        Args:
            bev_features: (B, in_channels, Y, X) - variable aspect ratio
            img_features: (B*S, feat2d, H/8, W/8)
            flip_indices: Optional flip augmentation

        Returns:
            Predictions at original (Y, X) resolution
        """
        B, C, Y_orig, X_orig = bev_features.shape

        # Step 1: Normalize to canonical aspect ratio
        bev_normalized = F.adaptive_avg_pool2d(bev_features, self.canonical_size)  # B, C, 128, 128

        # Step 2: Process with CNN (learns on consistent aspect ratio)
        x = self.conv1(bev_normalized)  # B, 256, 128, 128
        x = self.conv2(x)  # B, 256, 128, 128
        x = self.conv3(x)  # B, 128, 128, 128

        # Step 3: Upsample predictions back to original resolution
        center = self.center_head(x)  # B, n_classes, 128, 128
        center = F.interpolate(center, size=(Y_orig, X_orig), mode='bilinear', align_corners=False)

        offset = self.offset_head(x)  # B, 4, 128, 128
        offset = F.interpolate(offset, size=(Y_orig, X_orig), mode='bilinear', align_corners=False)

        size = self.size_head(x)  # B, 2, 128, 128
        size = F.interpolate(size, size=(Y_orig, X_orig), mode='bilinear', align_corners=False)

        rot = self.rot_head(x)  # B, 16, 128, 128
        rot = F.interpolate(rot, size=(Y_orig, X_orig), mode='bilinear', align_corners=False)

        # Handle augmentation flips
        if flip_indices is not None:
            flip1_index, flip2_index = flip_indices
            center[flip1_index] = torch.flip(center[flip1_index], [-1])
            center[flip2_index] = torch.flip(center[flip2_index], [-3])
            offset[flip1_index] = torch.flip(offset[flip1_index], [-1])
            offset[flip2_index] = torch.flip(offset[flip2_index], [-3])

        # Image center prediction (no normalization needed)
        img_center = self.img_center_head(img_features)

        return {
            'bev_raw': bev_features,  # Original resolution for temporal cache
            'instance_center': center,  # Upsampled to Y_orig, X_orig
            'instance_offset': offset,
            'instance_size': size,
            'instance_rot': rot,
            'img_center': img_center,
        }