import os
import os.path as osp
import torch
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import torch.nn.functional as F

from models import Segnet, MVDet, Liftnet, Bevformernet
from models.loss import FocalLoss, compute_rot_loss
from tracking.multitracker import JDETracker
from utils import vox, basic, decode
from evaluation.mod import modMetricsCalculator
from evaluation.mot_bev import mot_metrics
try:
    from tracking.multitracker_advanced import CascadedTracker
    from tracking.kalman_filter_extended import MotionModel
    ADVANCED_TRACKING_AVAILABLE = True
except ImportError as e:
    ADVANCED_TRACKING_AVAILABLE = False
    print(f"Warning: Advanced tracking not available: {e}")
    print("Falling back to standard tracking.")


class WorldTrackModel(pl.LightningModule):
    def __init__(
            self,
            model_name='segnet',
            encoder_name='res18',
            learning_rate=0.001,
            use_gcef=True,
            # Multi-resolution support - keep for backward compatibility
            resolution=(200, 4, 200),
            bounds=(-75, 75, -75, 75, -1, 5),
            # Multi-dataset mode
            multi_dataset_mode=False,
            resolutions_list=None,  # List of (Y, Z, X) tuples
            bounds_list=None,  # List of bounds tuples
            # Rest of params unchanged
            num_cameras=None,
            depth=(100, 2.0, 25),
            scene_centroid=(0.0, 0.0, 0.0),
            max_detections=60,
            conf_threshold=0.5,
            num_classes=1,
            use_temporal_cache=True,
            z_sign=1,
            feat2d_dim=128,
            use_advanced_tracking=True,
            tracking_motion_model='adaptive',
            use_trajectory_matching=True,
            use_bev_iou=True,
            cattle_size=30,
    ):
        super().__init__()

        # Store parameters
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.learning_rate = learning_rate
        self.max_detections = max_detections
        self.D, self.DMIN, self.DMAX = depth
        self.conf_threshold = conf_threshold
        self.use_advanced_tracking = use_advanced_tracking
        self.tracking_motion_model = tracking_motion_model
        self.use_trajectory_matching = use_trajectory_matching
        self.use_bev_iou = use_bev_iou
        self.cattle_size = cattle_size
        self.use_gcef = use_gcef

        # Multi-dataset support
        self.multi_dataset_mode = multi_dataset_mode

        if multi_dataset_mode:
            if resolutions_list is None or bounds_list is None:
                raise ValueError(
                    "multi_dataset_mode=True requires resolutions_list and bounds_list"
                )
            # Convert lists to tuples for consistent comparison
            self.resolutions_list = [tuple(r) for r in resolutions_list]
            self.bounds_list = [tuple(b) for b in bounds_list]

            # Use first resolution as default (for model initialization)
            self.resolution = self.resolutions_list[0]
            self.bounds = self.bounds_list[0]
            self.Y, self.Z, self.X = self.resolution

            print(f"\n{'=' * 60}")
            print("Multi-Dataset Mode ENABLED")
            print(f"Number of resolutions: {len(self.resolutions_list)}")
            for i, (res, bnd) in enumerate(zip(self.resolutions_list, self.bounds_list)):
                print(f"  Resolution {i}: Y={res[0]}, Z={res[1]}, X={res[2]}")
                print(f"    Bounds: {bnd}")
            print(f"{'=' * 60}\n")
        else:
            # Single dataset mode - also ensure tuples
            self.resolution = tuple(resolution) if not isinstance(resolution, tuple) else resolution
            self.bounds = tuple(bounds) if not isinstance(bounds, tuple) else bounds
            self.Y, self.Z, self.X = self.resolution
            self.resolutions_list = [self.resolution]
            self.bounds_list = [self.bounds]

        # Loss
        self.center_loss_fn = FocalLoss()

        # Temporal cache - now supports multiple resolutions
        self.use_temporal_cache = use_temporal_cache
        self.max_cache = 64
        self.temporal_cache_keys = [(-1, -1, -1) for _ in range(self.max_cache)]  # (seq, frame, res_idx)
        self.temporal_caches = {}  # {res_idx: cache_tensor}
        self.cache_hits = 0
        self.cache_misses = 0

        # Test tracking
        self.moda_gt_list, self.moda_pred_list = [], []
        self.mota_gt_list, self.mota_pred_list = [], []
        self.test_tracker_per_seq = {}
        self.trajectory_gt_data = {}
        self.trajectory_pred_data = {}
        self.mota_seq_names = []
        self.frame = 0

        # Multi-sequence support
        self.is_multi_sequence = False
        self.base = None

        # Create VoxelUtil for each resolution
        self.scene_centroid = torch.tensor(scene_centroid, device=self.device).reshape([1, 3])
        self.vox_utils = {}

        for idx, (res, bnd) in enumerate(zip(self.resolutions_list, self.bounds_list)):
            Y, Z, X = res
            self.vox_utils[idx] = vox.VoxelUtil(
                Y, Z, X,
                scene_centroid=self.scene_centroid,
                bounds=bnd,
                assert_cube=False
            )

        # Default vox_util (for backward compatibility)
        self.vox_util = self.vox_utils[0]

        # Model - initialized with default resolution
        num_cameras = None if num_cameras == 0 else num_cameras
        if model_name == 'segnet':
            self.model = Segnet(
                self.Y, self.Z, self.X,
                num_cameras=num_cameras,
                feat2d_dim=feat2d_dim,
                encoder_type=self.encoder_name,
                num_classes=num_classes,
                z_sign=z_sign,
                dynamic_resolution=multi_dataset_mode,  # Enable dynamic resolution
                use_gcef=self.use_gcef
            )
        elif model_name == 'liftnet':
            self.model = Liftnet(
                self.Y, self.Z, self.X,
                encoder_type=self.encoder_name,
                feat2d_dim=feat2d_dim,
                DMIN=self.DMIN,
                DMAX=self.DMAX,
                D=self.D,
                num_classes=num_classes,
                z_sign=z_sign,
                num_cameras=num_cameras,
                dynamic_resolution=multi_dataset_mode
            )
        elif model_name == 'bevformer':
            self.model = Bevformernet(
                self.Y, self.Z, self.X,
                feat2d_dim=feat2d_dim,
                encoder_type=self.encoder_name,
                num_classes=num_classes,
                z_sign=z_sign,
                dynamic_resolution=multi_dataset_mode
            )
        elif model_name == 'mvdet':
            self.model = MVDet(
                self.Y, self.Z, self.X,
                encoder_type=self.encoder_name,
                num_cameras=num_cameras,
                num_classes=num_classes,
                use_gcef=self.use_gcef
            )
        else:
            raise ValueError(f'Unknown model name {self.model_name}')

        self.save_hyperparameters()

    @staticmethod
    def _to_scalar(value):
        """
        Safely convert value to Python scalar (int/float)
        Handles: tensors, lists, numpy arrays, and scalars
        """
        if torch.is_tensor(value):
            return value.item()
        elif isinstance(value, (list, tuple)):
            return value[0] if len(value) > 0 else 0
        elif isinstance(value, np.ndarray):
            return value.item()
        else:
            return value

    def get_resolution_index(self, resolution):
        """Get index of resolution in resolutions_list"""
        # Ensure we're comparing tuples to tuples
        resolution_tuple = tuple(resolution) if not isinstance(resolution, tuple) else resolution

        for idx, res in enumerate(self.resolutions_list):
            if res == resolution_tuple:
                return idx

        # Provide helpful error message
        raise ValueError(
            f"Resolution {resolution_tuple} not found in resolutions_list.\n"
            f"Available resolutions: {self.resolutions_list}"
        )

    def get_vox_util_for_batch(self, item):
        """Get appropriate VoxelUtil for the current batch"""
        if not self.multi_dataset_mode:
            return self.vox_util

        # Get resolution from batch metadata
        if isinstance(item['resolution'], list):
            resolution = tuple(item['resolution'][0])  # All items in batch have same resolution
        else:
            resolution = tuple(item['resolution'])

        res_idx = self.get_resolution_index(resolution)
        return self.vox_utils[res_idx]

    def get_dataset_params(self, dataset_name):
        """
        Get resolution, bounds, and vox_util for a specific dataset
        Args:
            dataset_name: Name of the dataset (e.g., 'tracktacular', 'mmcows')
        Returns:
            tuple: (resolution, bounds, vox_util, res_idx)
        """
        if not self.multi_dataset_mode:
            # Single-dataset mode - return current model parameters
            return self.resolution, self.bounds, self.vox_util, 0

        # Multi-dataset mode - look up dataset config
        if hasattr(self.trainer.datamodule, 'dataset_configs'):
            for idx, config in enumerate(self.trainer.datamodule.dataset_configs):
                if config.name == dataset_name:
                    resolution = tuple(config.resolution)
                    bounds = tuple(config.bounds)
                    res_idx = self.get_resolution_index(resolution)
                    vox_util = self.vox_utils[res_idx]
                    return resolution, bounds, vox_util, res_idx

        # Fallback to default
        return self.resolution, self.bounds, self.vox_util, 0

    def setup(self, stage: Optional[str] = None):
        """Called at the beginning of fit/test to setup multi-sequence detection"""
        if stage == 'test' or stage is None:
            # === MULTI-SEQUENCE DETECTION ===
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'data_test'):
                if self.trainer.datamodule.data_test is not None:
                    # Handle both CombinedDataset (multi-dataset) and single dataset
                    if hasattr(self.trainer.datamodule.data_test, 'datasets'):
                        # Multi-dataset mode - CombinedDataset
                        self.is_multi_dataset_test = True
                        # Use the first dataset's base as default (for compatibility)
                        first_dataset_name = list(self.trainer.datamodule.data_test.datasets.keys())[0]
                        first_dataset = self.trainer.datamodule.data_test.datasets[first_dataset_name]
                        self.base = first_dataset.base
                        self.is_multi_sequence = hasattr(self.base, 'sequences')

                        # Build a combined sequence list from all datasets
                        self.all_sequences = []
                        for dataset_name, dataset in self.trainer.datamodule.data_test.datasets.items():
                            if hasattr(dataset.base, 'sequences'):
                                for seq_info in dataset.base.sequences:
                                    self.all_sequences.append({
                                        **seq_info,
                                        'dataset_name': dataset_name
                                    })

                        print(f"\n{'=' * 60}")
                        print("Multi-dataset testing mode detected!")
                        print(f"Number of datasets: {len(self.trainer.datamodule.data_test.datasets)}")
                        for dataset_name, dataset in self.trainer.datamodule.data_test.datasets.items():
                            if hasattr(dataset.base, 'sequences'):
                                print(f"\n  Dataset: {dataset_name}")
                                for seq_info in dataset.base.sequences:
                                    gt_status = "WITH GT" if seq_info.get('has_gt', True) else "NO GT"
                                    print(
                                        f"    - {seq_info['seq_name']}: {seq_info['num_frames']} frames [{gt_status}]")
                            else:
                                print(f"\n  Dataset: {dataset_name} (single sequence)")
                        print(f"\nTotal sequences across all datasets: {len(self.all_sequences)}")
                        print(f"{'=' * 60}\n")

                    elif hasattr(self.trainer.datamodule.data_test, 'base'):
                        # Single dataset mode
                        self.is_multi_dataset_test = False
                        self.base = self.trainer.datamodule.data_test.base
                        self.is_multi_sequence = hasattr(self.base, 'sequences')

                        if self.is_multi_sequence:
                            print(f"\n{'=' * 60}")
                            print("Multi-sequence mode detected!")
                            print(f"Number of sequences: {len(self.base.sequences)}")
                            for seq_info in self.base.sequences:
                                gt_status = "WITH GT" if seq_info.get('has_gt', True) else "NO GT"
                                print(f"  - {seq_info['seq_name']}: {seq_info['num_frames']} frames [{gt_status}]")
                            print(f"{'=' * 60}\n")
                        else:
                            print("\nSingle-sequence mode\n")
                    else:
                        # Fallback
                        self.is_multi_dataset_test = False
                        self.base = None
                        self.is_multi_sequence = False
                        print("\nUnknown dataset structure\n")

            # === RESET TRACKERS AND TRAJECTORY DATA ===
            self.test_tracker_per_seq = {}
            self.trajectory_gt_data = {}
            self.trajectory_pred_data = {}

            # Print tracking configuration
            if self.use_advanced_tracking:
                if ADVANCED_TRACKING_AVAILABLE:
                    print("[Setup] Advanced tracking initialized")
                    print(f"  Motion model: {self.tracking_motion_model}")
                    print(f"  Trajectory matching: {self.use_trajectory_matching}")
                    print(f"  BEV IoU: {self.use_bev_iou}")
                    print(f"  Cattle size: {self.cattle_size} cm\n")
                else:
                    print("[Setup] Advanced tracking requested but not available")
                    print("  Falling back to standard tracking\n")
                    self.use_advanced_tracking = False
            else:
                print("[Setup] Standard tracking initialized\n")

    def forward(self, item):
        """
        Forward pass with dynamic resolution support

        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        ref_T_global: (B,4,4)
        """
        # Get resolution for this batch
        if self.multi_dataset_mode:
            if isinstance(item['resolution'], list):
                resolution = tuple(item['resolution'][0])
            else:
                resolution = tuple(item['resolution'])
            res_idx = self.get_resolution_index(resolution)
            Y, Z, X = resolution
            vox_util = self.vox_utils[res_idx]
        else:
            Y, Z, X = self.Y, self.Z, self.X
            res_idx = 0
            vox_util = self.vox_util

        # Normalize sequence_num and frame to simple lists
        sequence_nums = item['sequence_num']
        frames = item['frame']

        # Convert to list of integers
        if torch.is_tensor(sequence_nums):
            sequence_nums = sequence_nums.cpu().tolist()
            if not isinstance(sequence_nums, list):  # Handle single-element tensor
                sequence_nums = [sequence_nums]
        elif not isinstance(sequence_nums, list):
            sequence_nums = [sequence_nums]

        # Ensure all elements are integers
        sequence_nums = [int(x) for x in sequence_nums]

        if torch.is_tensor(frames):
            frames = frames.cpu().tolist()
            if not isinstance(frames, list):  # Handle single-element tensor
                frames = [frames]
        elif not isinstance(frames, list):
            frames = [frames]

        # Ensure all elements are integers
        frames = [int(x) for x in frames]

        # Load previous BEV features from cache (resolution-aware)
        prev_bev = self.load_cache(sequence_nums, frames, res_idx)

        # Forward pass through model
        output = self.model(
            rgb_cams=item['img'],
            pix_T_cams=item['intrinsic'],
            cams_T_global=item['extrinsic'],
            ref_T_global=item['ref_T_global'],
            vox_util=vox_util,
            prev_bev=prev_bev,
            Y=Y,  # Pass resolution to model
            Z=Z,
            X=X,
        )

        # Store current BEV features in cache (resolution-aware)
        if self.use_temporal_cache:
            self.store_cache(
                sequence_nums,
                frames,
                output['bev_raw'].clone().detach(),
                res_idx
            )

        return output

    def get_sequence_info(self, seq_id):
        """
        Get sequence information by sequence ID
        Returns: (seq_name, has_gt, dataset_name)
        """
        # Multi-dataset mode
        if hasattr(self, 'is_multi_dataset_test') and self.is_multi_dataset_test:
            for seq_info in self.all_sequences:
                if seq_info['seq_id'] == seq_id:
                    return (
                        seq_info['seq_name'],
                        seq_info.get('has_gt', True),
                        seq_info['dataset_name']
                    )
            return f"seq_{seq_id}", True, "unknown"

        # Single dataset, multi-sequence mode
        if self.is_multi_sequence and hasattr(self, 'base') and self.base is not None:
            try:
                seq_idx = self.base.seq_id_to_index[seq_id]
                seq_info = self.base.sequences[seq_idx]
                return (
                    seq_info['seq_name'],
                    seq_info.get('has_gt', True),
                    getattr(self.base, '__name__', 'unknown')
                )
            except (KeyError, IndexError, AttributeError):
                pass

        # Fallback
        return f"seq_{seq_id}", True, "unknown"

    def load_cache(self, sequence_nums, frames, res_idx):
        """
        Load cached BEV features (resolution-aware).

        Args:
            sequence_nums: List of sequence IDs (length B)
            frames: List of frame numbers (length B)
            res_idx: Resolution index

        Returns:
            Cached BEV features or None if cache miss
        """
        if not self.use_temporal_cache or res_idx not in self.temporal_caches:
            return None

        B = len(frames)
        idx = []

        for seq_num, frame in zip(sequence_nums, frames):
            seq_num = int(seq_num)
            frame = int(frame)

            # Look for the previous frame in this sequence with this resolution
            target_key = (seq_num, frame - 1, res_idx)

            # Search for matching cache entry
            found = False
            for cache_idx, cache_key in enumerate(self.temporal_cache_keys):
                if cache_key == target_key:
                    idx.append(cache_idx)
                    found = True
                    self.cache_hits += 1
                    break

            if not found:
                self.cache_misses += 1
                return None  # Cache miss

        # Verify cached features match current resolution
        if len(idx) == B:
            cached_bev = self.temporal_caches[res_idx][idx]

            # Check if resolution matches (should match since we use res_idx)
            # This is a safety check
            return cached_bev
        else:
            return None

    def store_cache(self, sequence_nums, frames, bev_feat, res_idx):
        """
        Store BEV features in cache (resolution-aware)

        Args:
            sequence_nums: List of sequence IDs (length B)
            frames: List of frame numbers (length B)
            bev_feat: BEV features to cache (B, C, Y, X)
            res_idx: Resolution index
        """
        if not self.use_temporal_cache:
            return

        # Initialize cache for this resolution if needed
        if res_idx not in self.temporal_caches:
            shape = list(bev_feat.shape)
            shape[0] = self.max_cache
            self.temporal_caches[res_idx] = torch.zeros(
                shape, device=bev_feat.device, dtype=bev_feat.dtype
            )

        for seq_num, frame, feat in zip(sequence_nums, frames, bev_feat):
            seq_num = int(seq_num)
            frame = int(frame)
            cache_key = (seq_num, frame, res_idx)

            # Check if this key already exists in cache
            existing_idx = None
            for cache_idx, existing_key in enumerate(self.temporal_cache_keys):
                if existing_key == cache_key:
                    existing_idx = cache_idx
                    break

            if existing_idx is not None:
                # Update existing entry
                cache_slot = existing_idx
            else:
                # Find an empty slot or use random replacement
                empty_idx = None
                for cache_idx, existing_key in enumerate(self.temporal_cache_keys):
                    if existing_key == (-1, -1, -1):
                        empty_idx = cache_idx
                        break

                if empty_idx is not None:
                    cache_slot = empty_idx
                else:
                    # Random replacement
                    cache_slot = torch.randint(0, self.max_cache, (1,)).item()

            # Store feature and key
            self.temporal_caches[res_idx][cache_slot] = feat
            self.temporal_cache_keys[cache_slot] = cache_key

    def clear_cache_for_resolution(self, res_idx):
        """
        Clear temporal cache when switching between datasets with different resolutions.

        This prevents using cached BEV features from one resolution in another.
        Called automatically when resolution changes are detected.
        """
        if res_idx in self.temporal_caches:
            self.temporal_caches[res_idx].zero_()

        # Clear cache keys for this resolution
        for i in range(self.max_cache):
            cache_key = self.temporal_cache_keys[i]
            if cache_key[2] == res_idx:  # (seq_num, frame, res_idx)
                self.temporal_cache_keys[i] = (-1, -1, -1)

    def on_train_epoch_start(self):
        """Reset cache statistics at the start of each epoch"""
        self.cache_hits = 0
        self.cache_misses = 0

    def on_train_epoch_end(self):
        """Log cache statistics at the end of each epoch"""
        if self.use_temporal_cache and (self.cache_hits + self.cache_misses) > 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
            print(f"\n[Temporal Cache] Hits: {self.cache_hits}, Misses: {self.cache_misses}, Hit Rate: {hit_rate:.2%}")
            self.log('train/cache_hit_rate', hit_rate, on_epoch=True, sync_dist=True)

    def on_validation_epoch_start(self):
        """Clear cache at the start of validation"""
        if self.use_temporal_cache:
            self.temporal_cache_keys = [(-1, -1, -1) for _ in range(self.max_cache)]
            for res_idx in self.temporal_caches:
                self.temporal_caches[res_idx].zero_()

    def on_test_start(self):
        """Clear cache at the start of testing"""
        if self.use_temporal_cache:
            self.temporal_cache_keys = [(-1, -1, -1) for _ in range(self.max_cache)]
            for res_idx in self.temporal_caches:
                self.temporal_caches[res_idx].zero_()

    def loss(self, target, output):
        """
        Compute multi-task loss with uncertainty weighting.

        Handles spatial dimension mismatch when model resolution != data resolution
        """
        # ========================================
        # EXTRACT PREDICTIONS AND TARGETS
        # ========================================
        center_e = output['instance_center']
        offset_e = output['instance_offset']
        size_e = output['instance_size']
        rot_e = output['instance_rot']
        center_img_e = output['img_center']

        valid_g = target['valid_bev']
        center_g = target['center_bev']
        offset_g = target['offset_bev']

        # Handle spatial dimension mismatch
        # Check tensor dimensions and resize if needed
        needs_resize = False

        if len(center_g.shape) == 4:  # (B, C, H, W) - 2D BEV tensors
            B, C_g, H_gt, W_gt = center_g.shape
            _, C_e, H_pred, W_pred = center_e.shape
            needs_resize = (H_gt != H_pred) or (W_gt != W_pred)

            if needs_resize:
                # Use 2D pooling for 2D BEV tensors
                center_g = F.adaptive_max_pool2d(center_g, (H_pred, W_pred))
                offset_g = F.adaptive_avg_pool2d(offset_g, (H_pred, W_pred))
                valid_g = F.adaptive_max_pool2d(valid_g.float(), (H_pred, W_pred))

                # Resize size and rotation targets if they exist
                if 'size_bev' in target:
                    size_g = target['size_bev']
                    if len(size_g.shape) == 4 and size_g.shape[2] == H_gt:
                        target['size_bev'] = F.adaptive_avg_pool2d(size_g, (H_pred, W_pred))

                if 'rotbin_bev' in target:
                    rotbin_g = target['rotbin_bev']
                    if len(rotbin_g.shape) == 4 and rotbin_g.shape[2] == H_gt:
                        target['rotbin_bev'] = F.adaptive_max_pool2d(rotbin_g.float(), (H_pred, W_pred)).long()

                if 'rotres_bev' in target:
                    rotres_g = target['rotres_bev']
                    if len(rotres_g.shape) == 4 and rotres_g.shape[2] == H_gt:
                        target['rotres_bev'] = F.adaptive_avg_pool2d(rotres_g, (H_pred, W_pred))

        elif len(center_g.shape) == 5:  # (B, C, D, H, W) - 3D voxel tensors
            B, C_g, D_gt, H_gt, W_gt = center_g.shape
            _, C_e, D_pred, H_pred, W_pred = center_e.shape
            needs_resize = (D_gt != D_pred) or (H_gt != H_pred) or (W_gt != W_pred)

            if needs_resize:
                # Use 3D pooling for 3D voxel tensors
                center_g = F.adaptive_max_pool3d(center_g, (D_pred, H_pred, W_pred))
                offset_g = F.adaptive_avg_pool3d(offset_g, (D_pred, H_pred, W_pred))
                valid_g = F.adaptive_max_pool3d(valid_g.float(), (D_pred, H_pred, W_pred))

                # Resize size and rotation targets if they exist
                if 'size_bev' in target:
                    size_g = target['size_bev']
                    if len(size_g.shape) == 5 and size_g.shape[2] == D_gt:
                        target['size_bev'] = F.adaptive_avg_pool3d(size_g, (D_pred, H_pred, W_pred))

                if 'rotbin_bev' in target:
                    rotbin_g = target['rotbin_bev']
                    if len(rotbin_g.shape) == 5 and rotbin_g.shape[2] == D_gt:
                        target['rotbin_bev'] = F.adaptive_max_pool3d(rotbin_g.float(), (D_pred, H_pred, W_pred)).long()

                if 'rotres_bev' in target:
                    rotres_g = target['rotres_bev']
                    if len(rotres_g.shape) == 5 and rotres_g.shape[2] == D_gt:
                        target['rotres_bev'] = F.adaptive_avg_pool3d(rotres_g, (D_pred, H_pred, W_pred))

        B, S = target['center_img'].shape[:2]
        center_img_g = basic.pack_seqdim(target['center_img'], B)

        # ========================================
        # COMPUTE RAW LOSSES
        # ========================================
        # BEV center heatmap (per-sample normalized FocalLoss)
        center_loss = self.center_loss_fn(basic.sigmoid(center_e), center_g)

        # Offset regression (masked L1)
        offset_loss = torch.abs(offset_e[:, :2] - offset_g[:, :2]).sum(dim=1, keepdim=True)
        offset_loss = basic.reduce_masked_mean(offset_loss, valid_g)

        # Temporal offset (masked Smooth L1)
        tracking_loss = torch.nn.functional.smooth_l1_loss(
            offset_e[:, 2:], offset_g[:, 2:], reduction='none'
        ).sum(dim=1, keepdim=True)
        tracking_loss = basic.reduce_masked_mean(tracking_loss, valid_g)

        # Size and rotation (if available)
        if 'size_bev' in target:
            size_g = target['size_bev']
            rotbin_g = target['rotbin_bev']
            rotres_g = target['rotres_bev']

            size_loss = torch.abs(size_e - size_g).sum(dim=1, keepdim=True)
            size_loss = basic.reduce_masked_mean(size_loss, valid_g)
            rot_loss = compute_rot_loss(rot_e, rotbin_g, rotres_g, valid_g)
        else:
            size_loss = torch.tensor(0., device=center_e.device)
            rot_loss = torch.tensor(0., device=center_e.device)

        # Image center heatmap (per-sample normalized FocalLoss, averaged over cameras)
        center_img_loss = self.center_loss_fn(basic.sigmoid(center_img_e), center_img_g) / S

        # Apply scale factors
        center_loss_scaled = 10.0 * center_loss
        offset_loss_scaled = 10.0 * offset_loss

        # Uncertainty weighting
        center_loss_weighted = torch.exp(-self.model.center_weight) * center_loss_scaled
        center_uncertainty_reg = self.model.center_weight

        offset_loss_weighted = torch.exp(-self.model.offset_weight) * offset_loss_scaled
        offset_uncertainty_reg = self.model.offset_weight

        tracking_loss_weighted = torch.exp(-self.model.tracking_weight) * tracking_loss
        tracking_uncertainty_reg = self.model.tracking_weight

        size_loss_weighted = torch.exp(-self.model.size_weight) * size_loss
        size_uncertainty_reg = self.model.size_weight

        rot_loss_weighted = torch.exp(-self.model.rot_weight) * rot_loss
        rot_uncertainty_reg = self.model.rot_weight

        center_img_loss_weighted = torch.exp(-self.model.center_img_weight) * center_img_loss
        center_img_uncertainty_reg = self.model.center_img_weight

        # Aggregate losses
        loss_dict_raw = {
            'center_loss': center_loss.item() if isinstance(center_loss, torch.Tensor) else center_loss,
            'offset_loss': offset_loss.item() if isinstance(offset_loss, torch.Tensor) else offset_loss,
            'tracking_loss': tracking_loss.item() if isinstance(tracking_loss, torch.Tensor) else tracking_loss,
            'size_loss': size_loss.item() if isinstance(size_loss, torch.Tensor) else size_loss,
            'rot_loss': rot_loss.item() if isinstance(rot_loss, torch.Tensor) else rot_loss,
            'center_img': center_img_loss.item() if isinstance(center_img_loss, torch.Tensor) else center_img_loss,
        }

        loss_dict_weighted = {
            'center_loss_weighted': center_loss_weighted.item(),
            'offset_loss_weighted': offset_loss_weighted.item(),
            'tracking_loss_weighted': tracking_loss_weighted.item(),
            'size_loss_weighted': size_loss_weighted.item(),
            'rot_loss_weighted': rot_loss_weighted.item(),
            'center_img_weighted': center_img_loss_weighted.item(),
        }

        uncertainty_dict = {
            'center_uncertainty': torch.exp(self.model.center_weight).item(),
            'offset_uncertainty': torch.exp(self.model.offset_weight).item(),
            'tracking_uncertainty': torch.exp(self.model.tracking_weight).item(),
            'size_uncertainty': torch.exp(self.model.size_weight).item(),
            'rot_uncertainty': torch.exp(self.model.rot_weight).item(),
            'center_img_uncertainty': torch.exp(self.model.center_img_weight).item(),
        }

        total_loss = (
                center_loss_weighted + center_uncertainty_reg +
                offset_loss_weighted + offset_uncertainty_reg +
                tracking_loss_weighted + tracking_uncertainty_reg +
                size_loss_weighted + size_uncertainty_reg +
                rot_loss_weighted + rot_uncertainty_reg +
                center_img_loss_weighted + center_img_uncertainty_reg
        )

        loss_dict = {**loss_dict_raw, **loss_dict_weighted, **uncertainty_dict}

        return total_loss, loss_dict

    def training_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)
        total_loss, loss_dict = self.loss(target, output)

        B = item['img'].shape[0]

        # Log dataset-specific metrics (for multi-dataset debugging)
        if 'dataset_name' in item:
            dataset_name = item['dataset_name'][0] if isinstance(item['dataset_name'], list) else item['dataset_name']
            self.log(f'train/{dataset_name}/total_loss', total_loss, batch_size=B)

            # Log raw losses per dataset
            for key in ['center_loss', 'offset_loss', 'tracking_loss', 'size_loss', 'rot_loss', 'center_img']:
                if key in loss_dict:
                    self.log(f'train/{dataset_name}/{key}', loss_dict[key], batch_size=B)

        # Log overall metrics
        self.log('train_loss', total_loss, prog_bar=True, batch_size=B)

        # Separate logging for raw, weighted, and uncertainty losses
        for key, value in loss_dict.items():
            if 'uncertainty' in key:
                # Log uncertainty values separately (these are σ² = exp(s))
                self.log(f'train/uncertainty/{key}', value, batch_size=B)
            elif 'weighted' in key:
                # Log weighted losses
                self.log(f'train/weighted/{key}', value, batch_size=B)
            else:
                # Log raw losses
                self.log(f'train/raw/{key}', value, batch_size=B)

        return total_loss

    def validation_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        if batch_idx % 100 == 1:
            self.plot_data(target, output, item, batch_idx)  # Add 'item' parameter

        total_loss, loss_dict = self.loss(target, output)

        B = item['img'].shape[0]
        self.log('val_loss', total_loss, batch_size=B, sync_dist=True)
        self.log('val_center', loss_dict['center_loss'], batch_size=B, sync_dist=True)
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, batch_size=B, sync_dist=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        # Extract dataset information (handle both single and multi-dataset modes)
        if 'dataset_name' in item:
            # Multi-dataset mode
            if isinstance(item['dataset_name'], list):
                dataset_name = item['dataset_name'][0]
                dataset_id = item['dataset_id'][0].item() if torch.is_tensor(item['dataset_id'][0]) else \
                item['dataset_id'][0]
            else:
                dataset_name = item['dataset_name']
                dataset_id = item['dataset_id'].item() if torch.is_tensor(item['dataset_id']) else item['dataset_id']

            # Get dataset-specific parameters
            resolution, bounds, vox_util, res_idx = self.get_dataset_params(dataset_name)
        else:
            # Single-dataset mode (cross-dataset evaluation)
            dataset_name = getattr(self.trainer.datamodule, 'dataset', 'unknown')
            dataset_id = 0
            # Use model's current resolution/bounds
            resolution = self.resolution
            bounds = self.bounds
            vox_util = self.vox_util
            res_idx = 0

        Y, Z, X = resolution

        # Check if this batch has ground truth
        has_gt = item['has_gt'][0].item() if torch.is_tensor(item['has_gt'][0]) else item['has_gt'][0]

        # ========== VISUALIZATION ==========
        if not hasattr(self, 'vis_base_dir'):
            self.vis_base_dir = os.path.join(self.logger.log_dir, 'test_visualizations')
            os.makedirs(self.vis_base_dir, exist_ok=True)
            print(f"\n{'=' * 60}")
            print(f"Test visualizations will be saved to:")
            print(f"{self.vis_base_dir}")
            print(f"{'=' * 60}\n")

        # Get sequence information
        seq_num = int(item['sequence_num'][0].item())
        seq_name, has_gt_from_seq, _ = self.get_sequence_info(seq_num)

        # Create dataset-specific visualization directory
        dataset_vis_dir = os.path.join(self.vis_base_dir, dataset_name)
        seq_vis_dir = os.path.join(dataset_vis_dir, seq_name)

        # Generate visualizations (with or without GT)
        if batch_idx % 1 == 0:
            self.visualize_test_predictions(
                item, target, output, seq_vis_dir,
                has_gt=has_gt,
                vox_util=vox_util,
                bounds=bounds,
                resolution=resolution
            )

        # ========== DETECTION EVALUATION (only if GT exists) ==========
        if has_gt:
            center_e = output['instance_center']
            offset_e = output['instance_offset']
            size_e = output['instance_size']
            rot_e = output['instance_rot']

            xy_e, xy_prev_e, scores_e, classes_e, sizes_e, rzs_e = decode.decoder(
                center_e.sigmoid(), offset_e, size_e, rz_e=rot_e, K=self.max_detections
            )

            mem_xyz = torch.cat((xy_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
            ref_xy = vox_util.Mem2Ref(mem_xyz, Y, Z, X)[..., :2]
            mem_xyz_prev = torch.cat((xy_prev_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
            ref_xy_prev = vox_util.Mem2Ref(mem_xyz_prev, Y, Z, X)[..., :2]

            # Detection evaluation
            for frame, grid_gt, xy, score, seq_num_item in zip(
                    item['frame'], item['grid_gt'], ref_xy, scores_e, item['sequence_num']
            ):
                frame = int(frame.item())
                seq_num_item = int(seq_num_item.item())

                valid = score > self.conf_threshold
                gt_valid_mask = grid_gt.sum(1) != 0
                gt_boxes = grid_gt[gt_valid_mask]

                # Add ground truth - Format: [dataset_id, seq_num, frame, x, y]
                gt_list = []
                for gt_box in gt_boxes:
                    if len(gt_box) >= 2:
                        x, y = gt_box[0].item(), gt_box[1].item()
                        gt_list.append([dataset_id, seq_num_item, frame, x, y])
                self.moda_gt_list.extend(gt_list)

                # Add predictions
                pred_list = [
                    [dataset_id, seq_num_item, frame, x.item(), y.item()]
                    for x, y in xy[valid]
                ]
                self.moda_pred_list.extend(pred_list)

        # ========== TRACKING (always run, with or without GT) ==========
        center_e = output['instance_center']
        offset_e = output['instance_offset']
        size_e = output['instance_size']
        rot_e = output['instance_rot']

        xy_e, xy_prev_e, scores_e, classes_e, sizes_e, rzs_e = decode.decoder(
            center_e.sigmoid(), offset_e, size_e, rz_e=rot_e, K=self.max_detections
        )

        mem_xyz = torch.cat((xy_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy = vox_util.Mem2Ref(mem_xyz, Y, Z, X)[..., :2]
        mem_xyz_prev = torch.cat((xy_prev_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy_prev = vox_util.Mem2Ref(mem_xyz_prev, Y, Z, X)[..., :2]

        # Handle both tensor and list formats
        sequence_nums = item['sequence_num']
        frames = item['frame']

        if torch.is_tensor(sequence_nums):
            sequence_nums = sequence_nums.cpu().tolist()
        elif not isinstance(sequence_nums, list):
            sequence_nums = [sequence_nums]

        if torch.is_tensor(frames):
            frames = frames.cpu().tolist()
        elif not isinstance(frames, list):
            frames = [frames]

        for seq_num, frame, grid_gt, bev_det, bev_prev, score in zip(
                sequence_nums, frames, item['grid_gt'],
                ref_xy.cpu(), ref_xy_prev.cpu(), scores_e.cpu()
        ):
            frame = int(frame)
            seq_num = int(seq_num)

            # Get sequence name using helper method
            tracking_seq_name, _, _ = self.get_sequence_info(seq_num)

            # Initialize tracker for new sequence
            if seq_num not in self.test_tracker_per_seq:
                if self.use_advanced_tracking and ADVANCED_TRACKING_AVAILABLE:
                    motion_map = {
                        'constant_velocity': MotionModel.CONSTANT_VELOCITY,
                        'acceleration': MotionModel.CONSTANT_ACCELERATION,
                        'curvilinear': MotionModel.CURVILINEAR,
                        'adaptive': MotionModel.ADAPTIVE,
                    }
                    motion_model = motion_map.get(self.tracking_motion_model, MotionModel.ADAPTIVE)
                    self.test_tracker_per_seq[seq_num] = CascadedTracker(
                        conf_thres=self.conf_threshold,
                        track_buffer=30,
                        motion_model=motion_model,
                        use_trajectory_matching=self.use_trajectory_matching,
                        use_bev_iou=self.use_bev_iou,
                        cattle_size=self.cattle_size
                    )
                    print(f"\n[Tracker] Initialized CascadedTracker for {tracking_seq_name}")
                else:
                    self.test_tracker_per_seq[seq_num] = JDETracker(conf_thres=self.conf_threshold)
                    tracker_type = "JDETracker (fallback)" if self.use_advanced_tracking else "JDETracker"
                    print(f"\n[Tracker] Initialized {tracker_type} for {tracking_seq_name}")

                # Initialize trajectory data structures
                self.trajectory_gt_data[seq_num] = {}
                self.trajectory_pred_data[seq_num] = {}

            # Update tracker
            output_stracks = self.test_tracker_per_seq[seq_num].update(bev_det, bev_prev, score)

            # Add MOTA ground truth (only if GT exists)
            if has_gt:
                gt_valid_mask = grid_gt.sum(1) != 0
                gt_boxes = grid_gt[gt_valid_mask]
                gt_tracks = []
                for gt_box in gt_boxes:
                    if len(gt_box) >= 3:
                        x, y, i = gt_box[0].item(), gt_box[1].item(), gt_box[2].item()
                        # Format: [dataset_id, seq_num, frame, track_id, -1, -1, -1, -1, 1, x, y, -1]
                        self.mota_gt_list.append([dataset_id, seq_num, frame, i, -1, -1, -1, -1, 1, x, y, -1])
                        gt_tracks.append((x, y, i))

                        # Accumulate GT trajectory
                        track_id = int(i)
                        if track_id not in self.trajectory_gt_data[seq_num]:
                            self.trajectory_gt_data[seq_num][track_id] = []
                        self.trajectory_gt_data[seq_num][track_id].append((x, y, frame))

            # Add MOTA predictions
            pred_tracks = []
            for s in output_stracks:
                # Format: [dataset_id, seq_num, frame, track_id, -1, -1, -1, -1, score, x, y, -1]
                track_data = [dataset_id, seq_num, frame, s.track_id, -1, -1, -1, -1,
                              s.score.item()] + s.xy.tolist() + [-1]
                self.mota_pred_list.append(track_data)
                pred_tracks.append((s.xy[0].item(), s.xy[1].item(), s.track_id, s.score.item()))

                # Accumulate predicted trajectory
                track_id = s.track_id
                if track_id not in self.trajectory_pred_data[seq_num]:
                    self.trajectory_pred_data[seq_num][track_id] = []
                self.trajectory_pred_data[seq_num][track_id].append(
                    (s.xy[0].item(), s.xy[1].item(), frame, s.score.item())
                )


    def create_visualization_videos(self):
        """
        Create videos from saved visualization frames
        Requires: pip install opencv-python
        """
        try:
            import cv2
            import glob
            from PIL import Image

            if not hasattr(self, 'vis_base_dir'):
                return

            print(f"\n{'=' * 60}")
            print("Creating videos from visualizations...")

            for seq_dir in sorted(os.listdir(self.vis_base_dir)):
                seq_path = os.path.join(self.vis_base_dir, seq_dir)
                if not os.path.isdir(seq_path):
                    continue

                # ========== Create heatmap video ==========
                heatmap_pattern = os.path.join(seq_path, 'heatmap_frame_*.png')
                heatmap_frames = sorted(glob.glob(heatmap_pattern))

                if len(heatmap_frames) > 0:
                    print(f"\n  Processing {seq_dir} heatmaps ({len(heatmap_frames)} frames)...")

                    # Read first frame to get dimensions
                    first_img = cv2.imread(heatmap_frames[0])
                    if first_img is None:
                        print(f"    ERROR: Could not read {heatmap_frames[0]}")
                        continue

                    height, width = first_img.shape[:2]
                    print(f"    Frame size: {width}x{height}")

                    # Try multiple codec options
                    video_path = os.path.join(seq_path, f'{seq_dir}_heatmaps.mp4')

                    # Try different codecs in order of preference
                    codecs_to_try = [
                        ('avc1', '.mp4'),  # H.264 (best quality, widely supported)
                        ('mp4v', '.mp4'),  # MPEG-4
                        ('XVID', '.avi'),  # Xvid (fallback)
                    ]

                    video_created = False
                    for codec, ext in codecs_to_try:
                        if video_created:
                            break

                        video_path = os.path.join(seq_path, f'{seq_dir}_heatmaps{ext}')
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        video = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))

                        if not video.isOpened():
                            print(f"    Codec '{codec}' not available, trying next...")
                            continue

                        print(f"    Using codec: {codec}")

                        # Write all frames
                        frames_written = 0
                        for i, frame_path in enumerate(heatmap_frames):
                            img = cv2.imread(frame_path)

                            if img is None:
                                print(f"    WARNING: Could not read frame {i}: {frame_path}")
                                continue

                            # Ensure frame is exactly the right size
                            if img.shape[:2] != (height, width):
                                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

                            # Ensure BGR format
                            if len(img.shape) == 2:  # grayscale
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            elif img.shape[2] == 4:  # RGBA
                                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

                            video.write(img)
                            frames_written += 1

                        video.release()

                        # Verify the video was created successfully
                        if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
                            print(f"    ✓ Created: {video_path} ({frames_written} frames)")
                            video_created = True
                        else:
                            print(f"    Failed with codec '{codec}', trying next...")
                            if os.path.exists(video_path):
                                os.remove(video_path)

                    if not video_created:
                        print(f"    ERROR: Could not create heatmap video with any codec")

                # ========== Create detections video ==========
                det_pattern = os.path.join(seq_path, 'detections_frame_*.png')
                det_frames = sorted(glob.glob(det_pattern))

                if len(det_frames) > 0:
                    print(f"\n  Processing {seq_dir} detections ({len(det_frames)} frames)...")

                    # Read first frame to get dimensions
                    first_img = cv2.imread(det_frames[0])
                    if first_img is None:
                        print(f"    ERROR: Could not read {det_frames[0]}")
                        continue

                    height, width = first_img.shape[:2]
                    print(f"    Frame size: {width}x{height}")

                    video_created = False
                    for codec, ext in codecs_to_try:
                        if video_created:
                            break

                        video_path = os.path.join(seq_path, f'{seq_dir}_detections{ext}')
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        video = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))

                        if not video.isOpened():
                            print(f"    Codec '{codec}' not available, trying next...")
                            continue

                        print(f"    Using codec: {codec}")

                        frames_written = 0
                        for i, frame_path in enumerate(det_frames):
                            img = cv2.imread(frame_path)

                            if img is None:
                                print(f"    WARNING: Could not read frame {i}: {frame_path}")
                                continue

                            # Ensure frame is exactly the right size
                            if img.shape[:2] != (height, width):
                                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

                            # Ensure BGR format
                            if len(img.shape) == 2:
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            elif img.shape[2] == 4:
                                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

                            video.write(img)
                            frames_written += 1

                        video.release()

                        if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
                            print(f"    ✓ Created: {video_path} ({frames_written} frames)")
                            video_created = True
                        else:
                            print(f"    Failed with codec '{codec}', trying next...")
                            if os.path.exists(video_path):
                                os.remove(video_path)

                    if not video_created:
                        print(f"    ERROR: Could not create detections video with any codec")

                # ========== Create tracking video ==========
                tracking_dir = os.path.join(seq_path, 'tracking')
                if os.path.exists(tracking_dir):
                    tracking_pattern = os.path.join(tracking_dir, 'tracking_frame_*.png')
                    tracking_frames = sorted(glob.glob(tracking_pattern))

                    if len(tracking_frames) > 0:
                        print(f"\n  Processing {seq_dir} tracking ({len(tracking_frames)} frames)...")

                        first_img = cv2.imread(tracking_frames[0])
                        if first_img is None:
                            print(f"    ERROR: Could not read {tracking_frames[0]}")
                            continue

                        height, width = first_img.shape[:2]
                        print(f"    Frame size: {width}x{height}")

                        video_created = False
                        for codec, ext in codecs_to_try:
                            if video_created:
                                break

                            video_path = os.path.join(tracking_dir, f'{seq_dir}_tracking{ext}')
                            fourcc = cv2.VideoWriter_fourcc(*codec)
                            video = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))

                            if not video.isOpened():
                                print(f"    Codec '{codec}' not available, trying next...")
                                continue

                            print(f"    Using codec: {codec}")

                            frames_written = 0
                            for i, frame_path in enumerate(tracking_frames):
                                img = cv2.imread(frame_path)

                                if img is None:
                                    print(f"    WARNING: Could not read frame {i}: {frame_path}")
                                    continue

                                if img.shape[:2] != (height, width):
                                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

                                if len(img.shape) == 2:
                                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                                elif img.shape[2] == 4:
                                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

                                video.write(img)
                                frames_written += 1

                            video.release()

                            if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
                                print(f"    ✓ Created: {video_path} ({frames_written} frames)")
                                video_created = True
                            else:
                                print(f"    Failed with codec '{codec}', trying next...")
                                if os.path.exists(video_path):
                                    os.remove(video_path)

                        if not video_created:
                            print(f"    ERROR: Could not create tracking video with any codec")

            print(f"\n{'=' * 60}\n")

        except ImportError:
            print("\nOpenCV not available. Skipping video creation.")
            print("Install with: pip install opencv-python")
        except Exception as e:
            print(f"\nERROR creating videos: {str(e)}")
            import traceback
            traceback.print_exc()

    # Update on_test_epoch_end for per-sequence evaluation:
    def on_test_epoch_end(self):
        """Evaluate test results - supports multi-dataset testing"""
        import os
        from evaluation.mod import modMetricsCalculator
        from evaluation.mot_bev import mot_metrics

        # Save visualization summary
        if hasattr(self, 'vis_base_dir') and os.path.exists(self.vis_base_dir):
            print(f"\n{'=' * 60}")
            print(f"Test visualizations saved to:")
            print(f"{self.vis_base_dir}")
            for dataset_dir in sorted(os.listdir(self.vis_base_dir)):
                dataset_path = os.path.join(self.vis_base_dir, dataset_dir)
                if os.path.isdir(dataset_path):
                    print(f"\n  Dataset: {dataset_dir}")
                    for seq_dir in sorted(os.listdir(dataset_path)):
                        seq_path = os.path.join(dataset_path, seq_dir)
                        if os.path.isdir(seq_path):
                            num_frames = len([f for f in os.listdir(seq_path)
                                              if f.startswith('heatmap_') and f.endswith('.png')])
                            print(f"    {seq_dir}: {num_frames} frame visualizations")
            print(f"{'=' * 60}\n")

        # Check if we have any results
        if len(self.moda_pred_list) == 0:
            print("WARNING: No test results collected!")
            print("This might be because all test sequences lack ground truth.")
            return

        # Organize results by dataset
        all_res = np.array(self.moda_pred_list)  # [dataset_id, seq_num, frame, x, y]
        all_gt = np.array(self.moda_gt_list)
        all_mota_res = np.array(self.mota_pred_list)  # [dataset_id, seq_num, frame, track_id, ...]
        all_mota_gt = np.array(self.mota_gt_list)

        # Get unique dataset IDs
        dataset_ids = np.unique(all_res[:, 0].astype(int))

        # Get dataset names from datamodule
        if hasattr(self.trainer.datamodule, 'dataset_configs'):
            dataset_id_to_name = {i: cfg.name for i, cfg in enumerate(self.trainer.datamodule.dataset_configs)}
        else:
            dataset_id_to_name = {i: f"dataset_{i}" for i in dataset_ids}

        print(f"\n{'=' * 60}")
        print(f"Testing complete - {len(dataset_ids)} dataset(s)")
        for dataset_id in dataset_ids:
            print(f"  Dataset {dataset_id}: {dataset_id_to_name[dataset_id]}")
        print(f"{'=' * 60}\n")

        # Create output directory
        eval_dir = os.path.join(self.logger.log_dir, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)

        # ========== PER-DATASET EVALUATION ==========
        all_dataset_moda = []
        all_dataset_modp = []
        all_dataset_mota = []
        all_dataset_motp = []
        all_dataset_idf1 = []

        for dataset_id in dataset_ids:
            dataset_name = dataset_id_to_name[dataset_id]

            # Filter results for this dataset
            dataset_mask_res = all_res[:, 0] == dataset_id
            dataset_mask_gt = all_gt[:, 0] == dataset_id
            dataset_res = all_res[dataset_mask_res]
            dataset_gt = all_gt[dataset_mask_gt]

            dataset_mota_mask_res = all_mota_res[:, 0] == dataset_id
            dataset_mota_mask_gt = all_mota_gt[:, 0] == dataset_id
            dataset_mota_res = all_mota_res[dataset_mota_mask_res]
            dataset_mota_gt = all_mota_gt[dataset_mota_mask_gt]

            print(f"\n{'=' * 60}")
            print(f"EVALUATING DATASET: {dataset_name}")
            print(f"{'=' * 60}")
            print(f"Detections: {len(dataset_res)} predictions, {len(dataset_gt)} GT")
            print(f"Tracks: {len(dataset_mota_res)} predictions, {len(dataset_mota_gt)} GT")

            if len(dataset_res) == 0:
                print(f"WARNING: No predictions for {dataset_name}, skipping")
                continue

            # Get unique sequences in this dataset
            seq_ids = np.unique(dataset_res[:, 1].astype(int))

            dataset_moda_results = []
            dataset_modp_results = []

            # ========== DETECTION METRICS PER SEQUENCE ==========
            print(f"\n{'=' * 60}")
            print(f"DETECTION METRICS (MODA/MODP) - {dataset_name}")
            print(f"{'=' * 60}\n")

            for seq_id in seq_ids:
                # Get sequence name (handle multi-dataset mode)
                seq_name = f"seq_{seq_id}"  # fallback

                # Try to get from the current dataset being evaluated
                if hasattr(self, 'is_multi_dataset_test') and self.is_multi_dataset_test:
                    # Multi-dataset mode - look up in the specific dataset
                    if hasattr(self.trainer.datamodule, 'datasets_test'):
                        dataset_obj = self.trainer.datamodule.datasets_test.get(dataset_name)
                        if dataset_obj and hasattr(dataset_obj, 'base') and hasattr(dataset_obj.base, 'sequences'):
                            for seq_info in dataset_obj.base.sequences:
                                if seq_info['seq_id'] == seq_id:
                                    seq_name = seq_info['seq_name']
                                    break
                elif self.is_multi_sequence and hasattr(self, 'base') and self.base is not None:
                    # Single dataset mode
                    try:
                        seq_idx = self.base.seq_id_to_index[seq_id]
                        seq_name = self.base.sequences[seq_idx]['seq_name']
                    except (KeyError, IndexError, AttributeError):
                        seq_name = f"seq_{seq_id}"

                # Filter for this sequence
                seq_mask_res = dataset_res[:, 1] == seq_id
                seq_mask_gt = dataset_gt[:, 1] == seq_id
                seq_res = dataset_res[seq_mask_res][:, 2:5]  # [frame, x, y]
                seq_gt = dataset_gt[seq_mask_gt][:, 2:5]

                print(f"Evaluating {dataset_name}/{seq_name}")
                print(f"  Predictions: {len(seq_res)} detections")
                print(f"  Ground truth: {len(seq_gt)} detections")

                if len(seq_res) == 0:
                    print(f"  WARNING: No predictions, skipping\n")
                    continue

                # Save files
                res_fpath = os.path.join(eval_dir, f'{dataset_name}_{seq_name}_pred.txt')
                gt_fpath = os.path.join(eval_dir, f'{dataset_name}_{seq_name}_gt.txt')
                np.savetxt(res_fpath, seq_res, fmt='%d %.6f %.6f')
                np.savetxt(gt_fpath, seq_gt, fmt='%d %.6f %.6f')

                # Evaluate
                try:
                    recall, precision, MODA, MODP = modMetricsCalculator(res_fpath, gt_fpath)
                    print(f"  Results:")
                    print(f"    MODA: {MODA:.4f}")
                    print(f"    MODP: {MODP:.4f}")
                    print(f"    Recall: {recall:.4f}")
                    print(f"    Precision: {precision:.4f}\n")

                    self.log(f'test/{dataset_name}/{seq_name}/moda', MODA, on_epoch=True, sync_dist=True)
                    self.log(f'test/{dataset_name}/{seq_name}/modp', MODP, on_epoch=True, sync_dist=True)

                    dataset_moda_results.append(MODA)
                    dataset_modp_results.append(MODP)
                except Exception as e:
                    print(f"  ERROR: {str(e)}\n")
                    continue

            # Average detection metrics for this dataset
            if dataset_moda_results:
                avg_moda = np.mean(dataset_moda_results)
                avg_modp = np.mean(dataset_modp_results)

                print(f"{'=' * 60}")
                print(f"Average DETECTION metrics for {dataset_name}:")
                print(f"  MODA: {avg_moda:.4f}")
                print(f"  MODP: {avg_modp:.4f}")
                print(f"{'=' * 60}\n")

                self.log(f'test/{dataset_name}/avg_moda', avg_moda, on_epoch=True, sync_dist=True)
                self.log(f'test/{dataset_name}/avg_modp', avg_modp, on_epoch=True, sync_dist=True)

                all_dataset_moda.append(avg_moda)
                all_dataset_modp.append(avg_modp)

            # ========== TRACKING METRICS PER SEQUENCE ==========
            print(f"\n{'=' * 60}")
            print(f"TRACKING METRICS (MOTA/MOTP/IDF1) - {dataset_name}")
            print(f"{'=' * 60}\n")

            if len(dataset_mota_res) > 0:
                for seq_id in seq_ids:
                    # Get sequence name (handle multi-dataset mode)
                    seq_name = f"seq_{seq_id}"  # fallback

                    # Try to get from the current dataset being evaluated
                    if hasattr(self, 'is_multi_dataset_test') and self.is_multi_dataset_test:
                        # Multi-dataset mode - look up in the specific dataset
                        if hasattr(self.trainer.datamodule, 'datasets_test'):
                            dataset_obj = self.trainer.datamodule.datasets_test.get(dataset_name)
                            if dataset_obj and hasattr(dataset_obj, 'base') and hasattr(dataset_obj.base, 'sequences'):
                                for seq_info in dataset_obj.base.sequences:
                                    if seq_info['seq_id'] == seq_id:
                                        seq_name = seq_info['seq_name']
                                        break
                    elif self.is_multi_sequence and hasattr(self, 'base') and self.base is not None:
                        # Single dataset mode
                        try:
                            seq_idx = self.base.seq_id_to_index[seq_id]
                            seq_name = self.base.sequences[seq_idx]['seq_name']
                        except (KeyError, IndexError, AttributeError):
                            seq_name = f"seq_{seq_id}"

                    seq_mota_mask_res = dataset_mota_res[:, 1] == seq_id
                    seq_mota_mask_gt = dataset_mota_gt[:, 1] == seq_id
                    seq_mota_res = dataset_mota_res[seq_mota_mask_res].copy()
                    seq_mota_gt = dataset_mota_gt[seq_mota_mask_gt].copy()

                    # Set seq ID to 0 for single-sequence evaluation
                    seq_mota_res[:, 1] = 0
                    seq_mota_gt[:, 1] = 0
                    # Remove dataset_id column
                    seq_mota_res = seq_mota_res[:, 1:]
                    seq_mota_gt = seq_mota_gt[:, 1:]

                    print(f"Evaluating tracking: {dataset_name}/{seq_name}")
                    print(f"  Tracks: {len(seq_mota_res)} predictions, {len(seq_mota_gt)} GT")

                    if len(seq_mota_res) == 0:
                        print(f"  WARNING: No predictions, skipping\n")
                        continue

                    # Save
                    mota_res_fpath = os.path.join(eval_dir, f'{dataset_name}_{seq_name}_mota_pred.txt')
                    mota_gt_fpath = os.path.join(eval_dir, f'{dataset_name}_{seq_name}_mota_gt.txt')
                    np.savetxt(mota_res_fpath, seq_mota_res,
                               fmt='%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d', delimiter=',')
                    np.savetxt(mota_gt_fpath, seq_mota_gt,
                               fmt='%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%d', delimiter=',')

                    # Evaluate
                    try:
                        summary = mot_metrics(mota_res_fpath, mota_gt_fpath, scale=0.01)
                        overall = summary.loc['OVERALL']

                        print(f"  MOTA: {overall['mota'] * 100:.2f}%")
                        print(f"  MOTP: {overall['motp']:.4f}")
                        print(f"  IDF1: {overall['idf1'] * 100:.2f}%\n")

                        self.log(f'test/{dataset_name}/{seq_name}/mota', overall['mota'], on_epoch=True, sync_dist=True)
                        self.log(f'test/{dataset_name}/{seq_name}/motp', overall['motp'], on_epoch=True, sync_dist=True)
                        self.log(f'test/{dataset_name}/{seq_name}/idf1', overall['idf1'], on_epoch=True, sync_dist=True)
                    except Exception as e:
                        print(f"  ERROR: {str(e)}\n")
                        continue

                # Overall tracking for this dataset
                print(f"\nComputing overall tracking metrics for {dataset_name}...")
                dataset_mota_res_combined = dataset_mota_res.copy()
                dataset_mota_gt_combined = dataset_mota_gt.copy()

                # Renumber sequences starting from 0
                unique_seqs = np.unique(dataset_mota_res_combined[:, 1])
                for new_seq_id, old_seq_id in enumerate(unique_seqs):
                    mask_res = dataset_mota_res_combined[:, 1] == old_seq_id
                    mask_gt = dataset_mota_gt_combined[:, 1] == old_seq_id
                    dataset_mota_res_combined[mask_res, 1] = new_seq_id
                    dataset_mota_gt_combined[mask_gt, 1] = new_seq_id

                # Remove dataset_id column
                dataset_mota_res_combined = dataset_mota_res_combined[:, 1:]
                dataset_mota_gt_combined = dataset_mota_gt_combined[:, 1:]

                mota_res_all_fpath = os.path.join(eval_dir, f'{dataset_name}_all_sequences_mota_pred.txt')
                mota_gt_all_fpath = os.path.join(eval_dir, f'{dataset_name}_all_sequences_mota_gt.txt')

                np.savetxt(mota_res_all_fpath, dataset_mota_res_combined,
                           fmt='%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d', delimiter=',')
                np.savetxt(mota_gt_all_fpath, dataset_mota_gt_combined,
                           fmt='%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%d', delimiter=',')

                try:
                    summary_all = mot_metrics(mota_res_all_fpath, mota_gt_all_fpath, scale=0.01)
                    overall_all = summary_all.loc['OVERALL']

                    print(f"\n{'=' * 60}")
                    print(f"Overall TRACKING metrics for {dataset_name}:")
                    print(f"  MOTA: {overall_all['mota'] * 100:.2f}%")
                    print(f"  MOTP: {overall_all['motp']:.4f}")
                    print(f"  IDF1: {overall_all['idf1'] * 100:.2f}%")
                    print(f"{'=' * 60}\n")

                    self.log(f'test/{dataset_name}/overall_mota', overall_all['mota'], on_epoch=True, sync_dist=True)
                    self.log(f'test/{dataset_name}/overall_motp', overall_all['motp'], on_epoch=True, sync_dist=True)
                    self.log(f'test/{dataset_name}/overall_idf1', overall_all['idf1'], on_epoch=True, sync_dist=True)

                    all_dataset_mota.append(overall_all['mota'])
                    all_dataset_motp.append(overall_all['motp'])
                    all_dataset_idf1.append(overall_all['idf1'])
                except Exception as e:
                    print(f"ERROR: {str(e)}")

        # ========== OVERALL METRICS ACROSS ALL DATASETS ==========
        if len(all_dataset_moda) > 0:
            print(f"\n{'=' * 80}")
            print(f"OVERALL METRICS ACROSS ALL DATASETS")
            print(f"{'=' * 80}")
            print(f"\nDETECTION:")
            print(f"  Average MODA: {np.mean(all_dataset_moda):.4f}")
            print(f"  Average MODP: {np.mean(all_dataset_modp):.4f}")

            if len(all_dataset_mota) > 0:
                print(f"\nTRACKING:")
                print(f"  Average MOTA: {np.mean(all_dataset_mota) * 100:.2f}%")
                print(f"  Average MOTP: {np.mean(all_dataset_motp):.4f}")
                print(f"  Average IDF1: {np.mean(all_dataset_idf1) * 100:.2f}%")

            print(f"{'=' * 80}\n")

            # Log overall metrics
            self.log('test/overall_all_datasets_moda', np.mean(all_dataset_moda), on_epoch=True, sync_dist=True)
            self.log('test/overall_all_datasets_modp', np.mean(all_dataset_modp), on_epoch=True, sync_dist=True)
            if len(all_dataset_mota) > 0:
                self.log('test/overall_all_datasets_mota', np.mean(all_dataset_mota), on_epoch=True, sync_dist=True)
                self.log('test/overall_all_datasets_motp', np.mean(all_dataset_motp), on_epoch=True, sync_dist=True)
                self.log('test/overall_all_datasets_idf1', np.mean(all_dataset_idf1), on_epoch=True, sync_dist=True)

        # Create trajectory visualizations
        print(f"\n{'=' * 60}")
        print("Creating trajectory visualizations...")
        print(f"{'=' * 60}\n")

        if hasattr(self, 'vis_base_dir'):
            # Multi-dataset mode
            if hasattr(self, 'is_multi_dataset_test') and self.is_multi_dataset_test:
                for dataset_name in self.trainer.datamodule.dataset_configs:
                    # Get dataset-specific parameters
                    _, bounds, _, _ = self.get_dataset_params(dataset_name.name)

                    dataset_vis_dir = os.path.join(self.vis_base_dir, dataset_name.name)
                    dataset_obj = self.trainer.datamodule.data_test.datasets.get(dataset_name.name)

                    if dataset_obj and hasattr(dataset_obj.base, 'sequences'):
                        for seq_info in dataset_obj.base.sequences:
                            seq_id = seq_info['seq_id']
                            seq_name = seq_info['seq_name']
                            seq_vis_dir = os.path.join(dataset_vis_dir, seq_name)
                            has_gt = seq_info.get('has_gt', True)

                            print(
                                f"Generating trajectory plot for: {dataset_name.name}/{seq_name} ({'WITH GT' if has_gt else 'NO GT'})")
                            self.visualize_sequence_trajectories(seq_id, seq_name, seq_vis_dir, has_gt=has_gt,
                                                                 bounds=bounds)

            # Single dataset mode
            elif hasattr(self, 'base') and self.base is not None and hasattr(self.base, 'sequences'):
                for seq_info in self.base.sequences:
                    seq_id = seq_info['seq_id']
                    seq_name = seq_info['seq_name']
                    seq_vis_dir = os.path.join(self.vis_base_dir, seq_name)
                    has_gt = seq_info.get('has_gt', True)

                    print(f"Generating trajectory plot for: {seq_name} ({'WITH GT' if has_gt else 'NO GT'})")
                    self.visualize_sequence_trajectories(seq_id, seq_name, seq_vis_dir, has_gt=has_gt,
                                                         bounds=self.bounds)

        # Create videos
        self.create_visualization_videos()

        # Clear lists
        self.moda_gt_list.clear()
        self.moda_pred_list.clear()
        self.mota_gt_list.clear()
        self.mota_pred_list.clear()

    def plot_data(self, target, output, item, batch_idx=0):
        center_e = output['instance_center']
        center_g = target['center_bev']

        # Always index [0] first to get single element, then convert to scalar
        frame = self._to_scalar(item['frame'][0])
        seq_num = self._to_scalar(item['sequence_num'][0])

        # Get sequence name if available
        if self.is_multi_sequence and self.base is not None:
            seq_name = self.base.sequences[seq_num]['seq_name']
        else:
            seq_name = f'seq_{seq_num}'

        # Create output directory if it doesn't exist
        log_dir = self.trainer.log_dir if self.trainer.log_dir is not None else 'plots'
        os.makedirs(log_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.imshow(center_g[-1].amax(0).sigmoid().squeeze().cpu().numpy())
        ax2.imshow(center_e[-1].amax(0).sigmoid().squeeze().cpu().numpy())
        ax1.set_title('center_g (Ground Truth)')
        ax2.set_title('center_e (Prediction)')

        # Add frame information to the plot
        fig.suptitle(f'Sequence: {seq_name} | Frame: {frame:08d} | Batch: {batch_idx}',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout()

        # Try to use TensorBoard if available
        try:
            writer = self.logger.experiment
            if hasattr(writer, 'add_figure'):
                writer.add_figure(f'plot/{batch_idx}', fig, global_step=self.global_step)
            else:
                # Save to file if TensorBoard is not available
                figure_path = os.path.join(log_dir, f'plot_{batch_idx}_{self.global_step}_frame{frame:08d}.png')
                fig.savefig(figure_path)
                print(f"Saved validation plot: {figure_path}")
        except Exception as e:
            # Fallback to saving the figure to disk
            figure_path = os.path.join(log_dir, f'plot_{batch_idx}_{self.global_step}_frame{frame:08d}.png')
            fig.savefig(figure_path)
            print(f"Saved validation plot: {figure_path}")

        plt.close(fig)

    def visualize_test_predictions(self, item, target, output, save_dir, has_gt=True,
                                   vox_util=None, bounds=None, resolution=None):
        """
        Create comprehensive visualizations for test predictions and save to disk

        Args:
            item: Input data dict
            target: Ground truth dict
            output: Model predictions dict
            save_dir: Directory to save visualizations
            has_gt: Whether ground truth is available for this sequence
            vox_util: VoxelUtil object for this dataset (if None, uses self.vox_util)
            bounds: Bounds for this dataset (if None, uses self.bounds)
            resolution: Resolution (Y, Z, X) for this dataset (if None, uses self.Y, self.Z, self.X)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Use provided parameters or defaults
        if vox_util is None:
            vox_util = self.vox_util
        if bounds is None:
            bounds = self.bounds
        if resolution is None:
            Y, Z, X = self.Y, self.Z, self.X
        else:
            Y, Z, X = resolution

        # Decode predictions
        center_e = output['instance_center']
        offset_e = output['instance_offset']
        size_e = output['instance_size']
        rot_e = output['instance_rot']

        xy_e, xy_prev_e, scores_e, classes_e, sizes_e, rzs_e = decode.decoder(
            center_e.sigmoid(), offset_e, size_e, rz_e=rot_e, K=self.max_detections
        )

        mem_xyz = torch.cat((xy_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy = vox_util.Mem2Ref(mem_xyz, Y, Z, X)[..., :2]

        B = item['img'].shape[0]
        for b in range(B):
            frame = self._to_scalar(item['frame'][b])
            seq_num = self._to_scalar(item['sequence_num'][b])

            # Use get_sequence_info for reliable sequence name lookup
            seq_name, _, _ = self.get_sequence_info(seq_num)

            # Get predictions
            valid = scores_e[b] > self.conf_threshold
            pred_xy = ref_xy[b][valid].cpu().numpy()
            pred_scores = scores_e[b][valid].cpu().numpy()

            # Get ground truth (if available)
            if has_gt:
                grid_gt = item['grid_gt'][b]
                gt_valid_mask = grid_gt.sum(1) != 0
                gt_boxes = grid_gt[gt_valid_mask]
                gt_xy = gt_boxes[:, :2].cpu().numpy() if len(gt_boxes) > 0 else np.array([]).reshape(0, 2)
                gt_ids = gt_boxes[:, 2].cpu().numpy() if len(gt_boxes) > 0 else np.array([])
            else:
                gt_xy = np.array([]).reshape(0, 2)
                gt_ids = np.array([])

            os.makedirs(save_dir, exist_ok=True)

            # ============================================
            # IMAGE 1: HEATMAPS
            # ============================================
            if has_gt:
                # Show both GT and predicted heatmaps
                fig_heatmap = plt.figure(figsize=(16, 7))

                ax1 = fig_heatmap.add_subplot(1, 2, 1)
                center_g = target['center_bev'][b].amax(0).cpu().numpy()
                im1 = ax1.imshow(center_g, cmap='hot', origin='lower', aspect='equal')
                ax1.set_title('Ground Truth Center Heatmap', fontsize=14, fontweight='bold')
                ax1.set_xlabel('X (grid cells)', fontsize=12)
                ax1.set_ylabel('Y (grid cells)', fontsize=12)
                plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

                ax2 = fig_heatmap.add_subplot(1, 2, 2)
                center_p = center_e[b].amax(0).sigmoid().cpu().numpy()
                im2 = ax2.imshow(center_p, cmap='hot', origin='lower', aspect='equal')
                ax2.set_title('Predicted Center Heatmap', fontsize=14, fontweight='bold')
                ax2.set_xlabel('X (grid cells)', fontsize=12)
                ax2.set_ylabel('Y (grid cells)', fontsize=12)
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

                fig_heatmap.suptitle(
                    f'Sequence: {seq_name} | Frame: {frame} | GT: {len(gt_xy)} | Pred: {len(pred_xy)}',
                    fontsize=16, fontweight='bold'
                )
            else:
                # Show only predicted heatmap
                fig_heatmap = plt.figure(figsize=(10, 8))

                ax = fig_heatmap.add_subplot(1, 1, 1)
                center_p = center_e[b].amax(0).sigmoid().cpu().numpy()
                im = ax.imshow(center_p, cmap='hot', origin='lower', aspect='equal')
                ax.set_title('Predicted Center Heatmap', fontsize=14, fontweight='bold')
                ax.set_xlabel('X (grid cells)', fontsize=12)
                ax.set_ylabel('Y (grid cells)', fontsize=12)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

                fig_heatmap.suptitle(
                    f'Sequence: {seq_name} (NO GT) | Frame: {frame} | Pred: {len(pred_xy)}',
                    fontsize=16, fontweight='bold'
                )

            plt.tight_layout()
            heatmap_path = os.path.join(save_dir, f'heatmap_frame_{frame:06d}.png')
            fig_heatmap.savefig(heatmap_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig_heatmap)

            # ============================================
            # IMAGE 2: BEV DETECTIONS
            # ============================================
            fig_bev = plt.figure(figsize=(12, 10))
            ax = fig_bev.add_subplot(1, 1, 1)

            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlabel('X (world coordinates)', fontsize=12)
            ax.set_ylabel('Y (world coordinates)', fontsize=12)
            ax.set_facecolor('#f0f0f0')

            # Plot ground truth (if available)
            if has_gt and len(gt_xy) > 0:
                ax.scatter(gt_xy[:, 0], gt_xy[:, 1], c='lime', s=250, marker='o',
                           alpha=0.6, edgecolors='darkgreen', linewidths=3,
                           label=f'Ground Truth (n={len(gt_xy)})', zorder=4)

                for xy, pid in zip(gt_xy, gt_ids):
                    ax.text(xy[0], xy[1], f'{int(pid)}', fontsize=9, ha='center',
                            va='center', color='black', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

            # Plot predictions
            if len(pred_xy) > 0:
                scatter = ax.scatter(pred_xy[:, 0], pred_xy[:, 1], c=pred_scores,
                                     s=250, marker='X', alpha=0.8, cmap='plasma',
                                     edgecolors='navy', linewidths=3,
                                     vmin=self.conf_threshold, vmax=1.0,
                                     label=f'Predictions (n={len(pred_xy)})', zorder=5)
                cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Confidence Score', fontsize=11)

                for xy, score in zip(pred_xy, pred_scores):
                    offset_y = (bounds[3] - bounds[2]) * 0.02
                    ax.text(xy[0], xy[1] + offset_y, f'{score:.2f}', fontsize=8,
                            ha='center', va='bottom', color='red', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

            # Metrics text
            if has_gt:
                metrics_text = f'GT Detections: {len(gt_xy)}\nPredictions: {len(pred_xy)}'
            else:
                metrics_text = f'Predictions: {len(pred_xy)}\n(No GT Available)'

            if len(pred_xy) > 0:
                metrics_text += f'\nConf Range: [{pred_scores.min():.2f}, {pred_scores.max():.2f}]'
                metrics_text += f'\nMean Conf: {pred_scores.mean():.2f}'

            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

            title = f'Sequence: {seq_name} | Frame: {frame} | Bird\'s Eye View Detections'
            if not has_gt:
                title += ' (NO GT)'
            fig_bev.suptitle(title, fontsize=16, fontweight='bold')

            plt.tight_layout()
            bev_path = os.path.join(save_dir, f'detections_frame_{frame:06d}.png')
            fig_bev.savefig(bev_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig_bev)

            if frame % 10 == 0:
                gt_status = f"GT: {len(gt_xy)}" if has_gt else "NO GT"
                print(f"  [{seq_name}] Saved visualizations: frame {frame} ({gt_status})")

    def visualize_sequence_trajectories(self, seq_num, seq_name, save_dir, has_gt=True, bounds=None):
        """
        Create visualization showing complete trajectories for a sequence

        Args:
            seq_num: Sequence ID
            seq_name: Sequence name
            save_dir: Directory to save visualization
            has_gt: Whether this sequence has ground truth
            bounds: Bounds to use for plot limits (if None, uses self.bounds)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Use provided bounds or default
        if bounds is None:
            bounds = self.bounds

        if has_gt:
            # Two-panel plot (GT + Predictions)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))
            fig.suptitle(f'Complete Trajectories - {seq_name}', fontsize=18, fontweight='bold', y=0.98)

            # ========== Ground Truth Trajectories ==========
            ax1.set_xlim(bounds[0], bounds[1])
            ax1.set_ylim(bounds[2], bounds[3])
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax1.set_xlabel('X (world coordinates - cm)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Y (world coordinates - cm)', fontsize=14, fontweight='bold')
            ax1.set_facecolor('#f8f8f8')
            ax1.set_title('Ground Truth Trajectories', fontsize=16, fontweight='bold', pad=20)

            gt_trajectories = self.trajectory_gt_data.get(seq_num, {})
            if gt_trajectories:
                track_ids = sorted(gt_trajectories.keys())
                colors_gt = plt.cm.tab20(np.linspace(0, 1, len(track_ids)))
                if len(track_ids) > 20:
                    colors_gt = plt.cm.gist_rainbow(np.linspace(0, 1, len(track_ids)))

                for idx, track_id in enumerate(track_ids):
                    trajectory = sorted(gt_trajectories[track_id], key=lambda x: x[2])
                    if len(trajectory) < 2:
                        x, y, _ = trajectory[0]
                        ax1.scatter(x, y, c='green', s=150, marker='o',
                                    edgecolors='black', linewidths=2, alpha=0.8, zorder=5)
                    else:
                        xs = [p[0] for p in trajectory]
                        ys = [p[1] for p in trajectory]
                        ax1.plot(xs, ys, color=colors_gt[idx], linewidth=3, alpha=0.7,
                                 solid_capstyle='round', zorder=3)
                        ax1.scatter(xs[0], ys[0], c='green', s=200, marker='o',
                                    edgecolors='black', linewidths=2.5, alpha=0.9, zorder=5)
                        ax1.scatter(xs[-1], ys[-1], c='red', s=200, marker='s',
                                    edgecolors='black', linewidths=2.5, alpha=0.9, zorder=5)

                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                               markeredgecolor='black', markeredgewidth=2, markersize=12,
                               label='Start Points', linestyle='None'),
                    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                               markeredgecolor='black', markeredgewidth=2, markersize=12,
                               label='End Points', linestyle='None'),
                ]
                ax1.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1),
                           loc='upper left', fontsize=12,
                           framealpha=0.9, edgecolor='black', fancybox=True,
                           borderaxespad=0)

                total_points = sum(len(traj) for traj in gt_trajectories.values())
                stats_text = f'Tracks: {len(track_ids)}\nTotal Points: {total_points}\nAvg Length: {total_points / len(track_ids):.1f}'
                ax1.text(1.02, 0.5, stats_text, transform=ax1.transAxes,
                         fontsize=11, verticalalignment='center', horizontalalignment='left',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8,
                                   edgecolor='black', linewidth=1.5))
            else:
                ax1.text(0.5, 0.5, 'No Ground Truth Trajectories',
                         transform=ax1.transAxes, fontsize=16, ha='center', va='center',
                         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            # ========== Predicted Trajectories ==========
            ax2.set_xlim(bounds[0], bounds[1])
            ax2.set_ylim(bounds[2], bounds[3])
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax2.set_xlabel('X (world coordinates - cm)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Y (world coordinates - cm)', fontsize=14, fontweight='bold')
            ax2.set_facecolor('#f8f8f8')
            ax2.set_title('Predicted Trajectories', fontsize=16, fontweight='bold', pad=20)

            ax_pred = ax2
        else:
            # Single-panel plot (Predictions only)
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle(f'Complete Trajectories - {seq_name} (NO GT)', fontsize=18, fontweight='bold', y=0.98)

            ax_pred = fig.add_subplot(1, 1, 1)
            ax_pred.set_xlim(bounds[0], bounds[1])
            ax_pred.set_ylim(bounds[2], bounds[3])
            ax_pred.set_aspect('equal')
            ax_pred.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax_pred.set_xlabel('X (world coordinates - cm)', fontsize=14, fontweight='bold')
            ax_pred.set_ylabel('Y (world coordinates - cm)', fontsize=14, fontweight='bold')
            ax_pred.set_facecolor('#f8f8f8')
            ax_pred.set_title('Predicted Trajectories', fontsize=16, fontweight='bold', pad=20)

        # Plot predicted trajectories
        pred_trajectories = self.trajectory_pred_data.get(seq_num, {})
        if pred_trajectories:
            track_ids = sorted(pred_trajectories.keys())
            colors_pred = plt.cm.tab20(np.linspace(0, 1, len(track_ids)))
            if len(track_ids) > 20:
                colors_pred = plt.cm.gist_rainbow(np.linspace(0, 1, len(track_ids)))

            for idx, track_id in enumerate(track_ids):
                trajectory = sorted(pred_trajectories[track_id], key=lambda x: x[2])
                if len(trajectory) < 2:
                    x, y, _, score = trajectory[0]
                    ax_pred.scatter(x, y, c='green', s=150, marker='o',
                                    edgecolors='black', linewidths=2, alpha=0.8, zorder=5)
                else:
                    xs = [p[0] for p in trajectory]
                    ys = [p[1] for p in trajectory]
                    scores = [p[3] for p in trajectory]
                    avg_score = np.mean(scores)
                    alpha_val = 0.5 + (avg_score * 0.4)
                    ax_pred.plot(xs, ys, color=colors_pred[idx], linewidth=3,
                                 alpha=alpha_val,
                                 solid_capstyle='round', zorder=3)
                    ax_pred.scatter(xs[0], ys[0], c='green', s=200, marker='o',
                                    edgecolors='black', linewidths=2.5, alpha=0.9, zorder=5)
                    ax_pred.scatter(xs[-1], ys[-1], c='red', s=200, marker='s',
                                    edgecolors='black', linewidths=2.5, alpha=0.9, zorder=5)

            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                           markeredgecolor='black', markeredgewidth=2, markersize=12,
                           label='Start Points', linestyle='None'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                           markeredgecolor='black', markeredgewidth=2, markersize=12,
                           label='End Points', linestyle='None'),
            ]
            ax_pred.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1),
                           loc='upper left', fontsize=12,
                           framealpha=0.9, edgecolor='black', fancybox=True,
                           borderaxespad=0)

            total_points = sum(len(traj) for traj in pred_trajectories.values())
            all_scores = [score for traj in pred_trajectories.values() for _, _, _, score in traj]
            avg_conf = np.mean(all_scores) if all_scores else 0
            stats_text = f'Tracks: {len(track_ids)}\nTotal Points: {total_points}\nAvg Length: {total_points / len(track_ids):.1f}\nAvg Conf: {avg_conf:.3f}'
            ax_pred.text(1.02, 0.5, stats_text, transform=ax_pred.transAxes,
                         fontsize=11, verticalalignment='center', horizontalalignment='left',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8,
                                   edgecolor='black', linewidth=1.5))
        else:
            ax_pred.text(0.5, 0.5, 'No Predicted Trajectories',
                         transform=ax_pred.transAxes, fontsize=16, ha='center', va='center',
                         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.98, 0.96])
        plt.subplots_adjust(right=0.85)

        # Save
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f'{seq_name}_complete_trajectories.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"  ✓ Saved trajectory visualization: {output_path}")

    def visualize_tracking_results(self, seq_num, frame, gt_tracks, pred_tracks, save_dir):
        """
        Visualize tracking results with trajectory history

        Args:
            seq_num: Sequence number
            frame: Frame number
            gt_tracks: Ground truth tracks [(x, y, id), ...]
            pred_tracks: Predicted tracks [(x, y, id, score), ...]
            save_dir: Directory to save visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Circle
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Determine sequence name - CORRECTED for multi-sequence
        if self.is_multi_sequence and self.base is not None:
            seq_name = self.base.sequences[seq_num]['seq_name']
        else:
            seq_name = f'seq_{seq_num}'

        fig.suptitle(f'Tracking Results - {seq_name} | Frame: {frame}', fontsize=16, fontweight='bold')

        # Ground Truth Tracks
        ax1.set_xlim(self.bounds[0], self.bounds[1])
        ax1.set_ylim(self.bounds[2], self.bounds[3])
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Ground Truth Tracks (n={len(gt_tracks)})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (world coordinates)', fontsize=12)
        ax1.set_ylabel('Y (world coordinates)', fontsize=12)
        ax1.set_facecolor('#f5f5f5')

        # Plot GT tracks with different colors per ID
        if len(gt_tracks) > 0:
            unique_ids = list(set([t[2] for t in gt_tracks]))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_ids)))
            id_to_color = dict(zip(unique_ids, colors))

            for x, y, track_id in gt_tracks:
                color = id_to_color[track_id]
                ax1.scatter(x, y, c=[color], s=200, marker='o',
                            edgecolors='black', linewidths=2, alpha=0.7)
                ax1.text(x, y, f'{int(track_id)}', fontsize=10, ha='center',
                         va='center', color='white', fontweight='bold')

        # Predicted Tracks
        ax2.set_xlim(self.bounds[0], self.bounds[1])
        ax2.set_ylim(self.bounds[2], self.bounds[3])
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'Predicted Tracks (n={len(pred_tracks)})', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X (world coordinates)', fontsize=12)
        ax2.set_ylabel('Y (world coordinates)', fontsize=12)
        ax2.set_facecolor('#f5f5f5')

        # Plot predicted tracks
        if len(pred_tracks) > 0:
            unique_ids = list(set([t[2] for t in pred_tracks]))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_ids)))
            id_to_color = dict(zip(unique_ids, colors))

            for x, y, track_id, score in pred_tracks:
                color = id_to_color[track_id]
                # Size based on confidence
                size = 100 + score * 200
                ax2.scatter(x, y, c=[color], s=size, marker='X',
                            edgecolors='black', linewidths=2, alpha=0.7)
                ax2.text(x, y, f'{int(track_id)}\n{score:.2f}', fontsize=9,
                         ha='center', va='center', color='white', fontweight='bold')

        plt.tight_layout()

        # Save
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f'tracking_frame_{frame:06d}.png')
        fig.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            # Add gradient clipping
            "gradient_clip_val": 1.0,  # Clip gradients to max norm of 1.0
            "gradient_clip_algorithm": "norm"
        }


if __name__ == '__main__':
    from lightning.pytorch.cli import LightningCLI
    torch.set_float32_matmul_precision('medium')

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("model.resolution", "data.init_args.resolution")
            parser.link_arguments("model.bounds", "data.init_args.bounds")
            parser.link_arguments("trainer.accumulate_grad_batches", "data.init_args.accumulate_grad_batches")


    cli = MyLightningCLI(WorldTrackModel)
