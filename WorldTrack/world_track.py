# WorldTrack/world_track.py
import os
import os.path as osp
import time
from collections import defaultdict

import torch
import torch.nn as nn
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from models import Segnet, MVDet, Liftnet, Bevformernet
from models.loss import FocalLoss, reprojection_loss
from models.calibration_refinement import CalibrationRefinementModule
from tracking.multitracker import JDETracker
from utils import vox, basic, decode
from evaluation.mod import modMetricsCalculator
from evaluation.mot_bev import mot_metrics
from utils.id_assignment import (
    compute_ground_plane_homographies,
    assign_ids_to_detections,
    evaluate_2d_mot,
    visualize_2d_tracking,
)

# Optional YOLO import — with CLEAR diagnostic output
try:
    from ultralytics import YOLO as _YOLO
    HAS_YOLO = True
    print("[INIT] ✓ ultralytics imported successfully")
except ImportError:
    HAS_YOLO = False
    print("[INIT] ✗ ultralytics NOT installed — "             
          "2D tracking disabled. Fix: pip install ultralytics")

# Optional motmetrics
try:
    import motmetrics as _mm
    HAS_MOTMETRICS = True
    print("[INIT] ✓ motmetrics imported successfully")
except ImportError:
    HAS_MOTMETRICS = False
    print("[INIT] ✗ motmetrics NOT installed — "              
          "2D MOT eval disabled. Fix: pip install motmetrics")


class WorldTrackModel(pl.LightningModule):
    def __init__(
        self,
        model_name='segnet',
        encoder_name='res18',
        learning_rate=0.001,
        resolution=(200, 4, 200),
        bounds=(-75, 75, -75, 75, -1, 5),
        num_cameras=None,
        depth=(100, 2.0, 25),
        scene_centroid=(0.0, 0.0, 0.0),
        max_detections=60,
        conf_threshold=0.5,
        num_classes=1,
        use_temporal_cache=True,
        z_sign=1,
        feat2d_dim=128,
        # ── image auxiliary (training) ───────────────────────
        use_image_aux_loss: bool = False,
        img_aux_weight: float = 1.0,
        # ── 2D per-camera tracking (test) ────────────────────
        use_2d_tracking: bool = False,
        yolo_weights_path: str = (
            'yolo_weights/MmCows_CowsOnly/weights/best.pt'
        ),
        tau_score: float = 0.3,
        tau_match: float = 100.0,
        original_img_hw: tuple = (2800, 4480),
        # ── 2D matching geometry ────────────────────
        use_foot_point: bool = True,
        adaptive_tau: bool = True,
        tau_floor: float = 50.0,
        tau_fraction: float = 0.15,
        cost_gate_factor: float = 3.0,
        projection_margin: float = 0.02,
        save_2d_visualizations: bool = True,
        viz_every_n_frames: int = 10,
        # ── calibration refinement ───────────────────────────
        learn_calibration: bool = False,
        cal_refine_reg_weight: float = 1e-4,
        lr_calib_scale: float = 0.1,
        calibration_dir: str = "",
        reset_calibration_at_test: bool = False,
        # ── reprojection loss ────────────────────────────────
        use_reproj_loss: bool = True,
        reproj_weight: float = 0.1,
        min_obs_per_camera: int = 0,
        reproj_soft_gate: bool = False,
        reproj_soft_gate_temp: float = 1.0,
        reproj_loss_type: str = 'smooth_l1',
        reproj_robust_scale: float = 50.0,
        reproj_camera_weighting: str = 'proportional',
        calib_warmup_steps: int = 0,
        calib_warmup_mode: str = 'steps',
        calib_warmup_loss_threshold: float = 1.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.Y, self.Z, self.X = self.resolution
        self.bounds = bounds
        self.max_detections = max_detections
        self.D, self.DMIN, self.DMAX = depth
        self.conf_threshold = conf_threshold

        # Config
        self.use_image_aux_loss = use_image_aux_loss
        self.img_aux_weight = img_aux_weight
        self.use_2d_tracking = use_2d_tracking
        self.yolo_weights_path = yolo_weights_path
        self.tau_score = tau_score
        self.tau_match = tau_match
        self.original_img_hw = original_img_hw
        # 2D matching geometry
        self.use_foot_point = use_foot_point
        self.adaptive_tau = adaptive_tau
        self.tau_floor = tau_floor
        self.tau_fraction = tau_fraction
        self.cost_gate_factor = cost_gate_factor
        self.projection_margin = projection_margin
        self.save_2d_visualizations = save_2d_visualizations
        self.viz_every_n_frames = viz_every_n_frames

        # Calibration refinement
        self.learn_calibration = learn_calibration
        self.cal_refine_reg_weight = cal_refine_reg_weight
        self.lr_calib_scale = lr_calib_scale
        self.calibration_dir = calibration_dir
        self.reset_calibration_at_test = reset_calibration_at_test
        self.use_reproj_loss = use_reproj_loss
        self.reproj_weight = reproj_weight
        # I2: minimum observation gate
        self.min_obs_per_camera = min_obs_per_camera
        self.reproj_soft_gate = reproj_soft_gate
        self.reproj_soft_gate_temp = reproj_soft_gate_temp
        self.reproj_loss_type = reproj_loss_type
        self.reproj_robust_scale = reproj_robust_scale
        self.reproj_camera_weighting = reproj_camera_weighting
        self.calib_warmup_steps = calib_warmup_steps
        self.calib_warmup_mode = calib_warmup_mode
        self.calib_warmup_loss_threshold = calib_warmup_loss_threshold
        self._calib_unfrozen = False
        self._latest_center_loss = float('inf')
        # validation reprojection accumulators (Welford's online mean)
        self._val_reproj_n = None  # dict of cam → int
        self._val_reproj_sum = None  # dict of cam → float (base)
        self._val_reproj_sum_r = None  # dict of cam → float (refined)

        # Loss
        self.center_loss_fn = FocalLoss()

        # Temporal cache
        self.use_temporal_cache = use_temporal_cache
        self.max_cache = 32
        self.temporal_cache_frames = -2 * torch.ones(
            self.max_cache, dtype=torch.long
        )
        self.temporal_cache_seqs = -2 * torch.ones(
            self.max_cache, dtype=torch.long
        )
        self.temporal_cache = None

        # Test bookkeeping
        self.moda_gt_list, self.moda_pred_list = [], []
        self.mota_gt_list, self.mota_pred_list = [], []
        self.frame = 0
        self.test_tracker = JDETracker(conf_thres=self.conf_threshold)
        self._current_test_seq = -1
        self.inference_times = []
        self.test_start_time = None

        # 2D tracking accumulators
        self.cam2d_pred = defaultdict(list)
        self.cam2d_gt = defaultdict(list)
        self.yolo = None
        self._2d_frame_counter = 0

        # Model
        num_cameras = None if num_cameras == 0 else num_cameras
        _model_kwargs = dict(use_image_aux=use_image_aux_loss)

        if model_name == 'segnet':
            self.model = Segnet(
                self.Y, self.Z, self.X,
                num_cameras=num_cameras,
                feat2d_dim=feat2d_dim,
                encoder_type=self.encoder_name,
                num_classes=num_classes,
                z_sign=z_sign,
                **_model_kwargs,
            )
        elif model_name == 'liftnet':
            self.model = Liftnet(
                self.Y, self.Z, self.X,
                encoder_type=self.encoder_name,
                feat2d_dim=feat2d_dim,
                DMIN=self.DMIN, DMAX=self.DMAX, D=self.D,
                num_classes=num_classes,
                z_sign=z_sign,
                num_cameras=num_cameras,
                **_model_kwargs,
            )
        elif model_name == 'bevformer':
            self.model = Bevformernet(
                self.Y, self.Z, self.X,
                feat2d_dim=feat2d_dim,
                encoder_type=self.encoder_name,
                num_classes=num_classes,
                z_sign=z_sign,
                **_model_kwargs,
            )
        elif model_name == 'mvdet':
            self.model = MVDet(
                self.Y, self.Z, self.X,
                encoder_type=self.encoder_name,
                num_cameras=num_cameras,
                num_classes=num_classes,
                **_model_kwargs,
            )
        else:
            raise ValueError(f'Unknown model name {self.model_name}')

        # Calibration refinement module
        if self.learn_calibration:
            _num_cams = (num_cameras
                         if (num_cameras and num_cameras > 0)
                         else 4)
            self.cal_refine = CalibrationRefinementModule(
                num_cameras=_num_cams
            )
            # I6: freeze if warmup is requested
            if self.calib_warmup_steps > 0 or \
                    self.calib_warmup_mode == 'loss_threshold':
                for p in self.cal_refine.parameters():
                    p.requires_grad = False
                self._calib_unfrozen = False
            else:
                self._calib_unfrozen = True

        # Learned log-variance for reprojection loss (uncertainty balancing)
        if self.learn_calibration and self.use_reproj_loss:
            self.reproj_log_var = nn.Parameter(
                torch.tensor(0.0), requires_grad=True
            )

        self.scene_centroid = torch.tensor(
            scene_centroid, device=self.device
        ).reshape([1, 3])
        self.vox_util = vox.VoxelUtil(
            self.Y, self.Z, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds,
        )

        self.save_hyperparameters(ignore=[
            'use_2d_tracking',
            'tau_score',
            'tau_match',
            'original_img_hw',
            'save_2d_visualizations',
            'viz_every_n_frames',
            'reset_calibration_at_test',
            'calibration_dir',
            'use_foot_point',
            'adaptive_tau',
            'tau_floor',
            'tau_fraction',
            'cost_gate_factor',
            'projection_margin',
            'calib_warmup_steps',
            'calib_warmup_mode',
            'calib_warmup_loss_threshold',
        ])

    # ==================================================================
    # Forward + temporal cache
    # ==================================================================
    def forward(self, item):
        frames_cpu = item['frame'].cpu()
        seq_nums_cpu = item.get(
            'sequence_num', torch.zeros_like(item['frame'])
        ).cpu()
        prev_bev = self.load_cache(frames_cpu, seq_nums_cpu)

        # Apply calibration refinement if enabled
        if self.learn_calibration:
            refined_extrinsics = self.cal_refine(item['extrinsic'])
        else:
            refined_extrinsics = item['extrinsic']

        output = self.model(
            rgb_cams=item['img'],
            pix_T_cams=item['intrinsic'],
            cams_T_global=refined_extrinsics,
            ref_T_global=item['ref_T_global'],
            vox_util=self.vox_util,
            prev_bev=prev_bev,
        )

        if self.use_temporal_cache:
            self.store_cache(
                frames_cpu,
                output['bev_raw'].clone().detach(),
                seq_nums_cpu,
            )

        # Store refined extrinsics (detached) for test-time 2D tracking
        if self.learn_calibration:
            output['extrinsic_refined'] = refined_extrinsics.detach()

        return output

    def load_cache(self, frames, sequence_nums=None):
        if self.temporal_cache is None:
            return None
        if sequence_nums is None:
            sequence_nums = torch.zeros_like(frames)
        idx = []
        for frame, seq in zip(frames, sequence_nums):
            match = (
                (frame - 1 == self.temporal_cache_frames)
                & (seq == self.temporal_cache_seqs)
            ).nonzero(as_tuple=True)[0]
            if match.nelement() == 1:
                idx.append(match.item())
        if len(idx) != len(frames):
            return None
        return self.temporal_cache[idx]

    def store_cache(self, frames, bev_feat, sequence_nums=None):
        if sequence_nums is None:
            sequence_nums = torch.zeros_like(frames)
        if self.temporal_cache is None:
            shape = list(bev_feat.shape)
            shape[0] = self.max_cache
            self.temporal_cache = torch.zeros(
                shape, device=bev_feat.device, dtype=bev_feat.dtype
            )
        for frame, feat, seq in zip(frames, bev_feat, sequence_nums):
            i = (
                (frame - 1 == self.temporal_cache_frames)
                & (seq == self.temporal_cache_seqs)
            ).nonzero(as_tuple=True)[0]
            if i.nelement() == 0:
                i = (
                    self.temporal_cache_frames == -2
                ).nonzero(as_tuple=True)[0]
            if i.nelement() == 0:
                i = torch.randint(self.max_cache, (1, 1))
            self.temporal_cache[i[0]] = feat
            self.temporal_cache_frames[i[0]] = frame
            self.temporal_cache_seqs[i[0]] = seq

    # ==================================================================
    # Loss
    # ==================================================================
    def loss(self, target, output, item=None):
        center_e = output['instance_center']
        offset_e = output['instance_offset']
        center_img_e = output['img_center']

        valid_g = target['valid_bev']
        center_g = target['center_bev']
        offset_g = target['offset_bev']

        B, S = target['center_img'].shape[:2]
        center_img_g = basic.pack_seqdim(target['center_img'], B)

        # BEV losses
        center_loss = self.center_loss_fn(
            basic.sigmoid(center_e), center_g
        )
        offset_loss = torch.abs(
            offset_e[:, :2] - offset_g[:, :2]
        ).sum(dim=1, keepdim=True)
        offset_loss = basic.reduce_masked_mean(offset_loss, valid_g)

        tracking_loss = torch.nn.functional.smooth_l1_loss(
            offset_e[:, 2:], offset_g[:, 2:], reduction='none'
        ).sum(dim=1, keepdim=True)
        tracking_loss = basic.reduce_masked_mean(
            tracking_loss, valid_g
        )

        center_factor = 1 / torch.exp(self.model.center_weight)
        center_loss_weight = center_factor * center_loss
        center_uncertainty_loss = self.model.center_weight

        offset_factor = 1 / torch.exp(self.model.offset_weight)
        offset_loss_weight = offset_factor * offset_loss
        offset_uncertainty_loss = self.model.offset_weight

        tracking_factor = 1 / torch.exp(self.model.tracking_weight)
        tracking_loss_weight = tracking_factor * tracking_loss
        tracking_uncertainty_loss = self.model.tracking_weight

        center_img_loss = self.center_loss_fn(
            basic.sigmoid(center_img_e), center_img_g
        ) / S

        loss_dict = {
            'center_loss': 10 * center_loss,
            'offset_loss': 10 * offset_loss,
            'tracking_loss': tracking_loss,
            'center_img': center_img_loss,
        }

        loss_weight_dict = {
            'center_loss': 10 * center_loss_weight,
            'offset_loss': 10 * offset_loss_weight,
            'tracking_loss': tracking_loss_weight,
            'center_img': center_img_loss,
        }

        stats_dict = {
            'center_uncertainty_loss': center_uncertainty_loss,
            'offset_uncertainty_loss': offset_uncertainty_loss,
            'tracking_uncertainty_loss': tracking_uncertainty_loss,
        }

        # Image auxiliary losses
        if self.use_image_aux_loss and 'img_offset' in output:
            offset_img_g = basic.pack_seqdim(
                target['offset_img'], B
            )
            size_img_g = basic.pack_seqdim(
                target['size_img'], B
            )
            valid_img_g = basic.pack_seqdim(
                target['valid_img'], B
            )

            offset_img_loss = torch.abs(
                output['img_offset'] - offset_img_g
            ).sum(dim=1, keepdim=True)
            offset_img_loss = basic.reduce_masked_mean(
                offset_img_loss, valid_img_g
            ) / S

            size_img_loss = torch.abs(
                output['img_size'] - size_img_g
            ).sum(dim=1, keepdim=True)
            size_img_loss = basic.reduce_masked_mean(
                size_img_loss, valid_img_g
            ) / S

            w = self.img_aux_weight
            loss_dict['offset_img'] = w * offset_img_loss
            loss_dict['size_img'] = w * size_img_loss
            loss_weight_dict['offset_img'] = w * offset_img_loss
            loss_weight_dict['size_img'] = w * size_img_loss

        # Reprojection consistency loss
        if (self.learn_calibration and self.use_reproj_loss
                and item is not None and self._calib_unfrozen):
            refined_ext = self.cal_refine(item['extrinsic'])

            reproj = reprojection_loss(
                item['grid_gt'],
                item['img_gt_2d'],
                item['intrinsic_original'],
                refined_ext,
                item['worldcoord_from_worldgrid'],
                camera_weighting=self.reproj_camera_weighting,
                min_obs_per_camera=self.min_obs_per_camera,
                soft_gate=self.reproj_soft_gate,
                soft_gate_temp=self.reproj_soft_gate_temp,
                loss_type=self.reproj_loss_type,
                robust_scale=self.reproj_robust_scale,
            )

            # Uncertainty-weighted reprojection loss
            reproj_factor = 1.0 / torch.exp(self.reproj_log_var)
            loss_dict['reproj_loss'] = self.reproj_weight * reproj
            loss_weight_dict['reproj_loss'] = (
                self.reproj_weight * reproj_factor * reproj
            )
            stats_dict['reproj_uncertainty'] = self.reproj_log_var

        total_loss = (sum(loss_weight_dict.values())
                      + sum(stats_dict.values()))

        # ── L2 regularisation on calibration parameters ───────
        if self.learn_calibration:
            reg = (self.cal_refine.delta_r.pow(2).sum()
                   + self.cal_refine.delta_t.pow(2).sum())
            total_loss = total_loss + self.cal_refine_reg_weight * reg

        return total_loss, loss_dict

    # ==================================================================
    # Training / validation steps
    # ==================================================================
    def training_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)
        total_loss, loss_dict = self.loss(target, output, item)

        B = item['img'].shape[0]
        self.log('train_loss', total_loss, prog_bar=True, batch_size=B)
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, batch_size=B)

        # Log calibration drift
        if self.learn_calibration:
            for cam_idx, (dr_norm, dt_norm) in \
                    self.cal_refine.get_delta_norms().items():
                self.log(f'calib/cam{cam_idx}_delta_r_rad', dr_norm,
                         on_step=False, on_epoch=True, batch_size=B)
                self.log(f'calib/cam{cam_idx}_delta_t_world', dt_norm,
                         on_step=False, on_epoch=True, batch_size=B)
        # I6: track center loss for warmup
        if self.learn_calibration and not self._calib_unfrozen:
            self._latest_center_loss = (
                loss_dict['center_loss'].detach().item()
            )

        return total_loss

    def validation_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        if batch_idx % 100 == 1:
            self.plot_data(target, output, batch_idx)

        total_loss, loss_dict = self.loss(target, output, item)

        B = item['img'].shape[0]
        self.log('val_loss', total_loss, batch_size=B, sync_dist=True)
        self.log(
            'val_center', loss_dict['center_loss'],
            batch_size=B, sync_dist=True,
        )
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, batch_size=B, sync_dist=True)

        # Log calibration drift
        if self.learn_calibration:
            for cam_idx, (dr_norm, dt_norm) in \
                    self.cal_refine.get_delta_norms().items():
                self.log(f'calib/cam{cam_idx}_delta_r_rad', dr_norm,
                         on_step=False, on_epoch=True, batch_size=B)
                self.log(f'calib/cam{cam_idx}_delta_t_world', dt_norm,
                         on_step=False, on_epoch=True, batch_size=B)

        # accumulate per-camera reprojection errors
        if self.learn_calibration and self._val_reproj_n is not None:
            self._accumulate_reproj_diagnostics(item)

        return total_loss

    def on_validation_epoch_start(self):
        """I8: Reset per-camera reprojection accumulators."""
        if not self.learn_calibration:
            return
        S = self.cal_refine.num_cameras
        self._val_reproj_n = defaultdict(int)
        self._val_reproj_sum = defaultdict(float)
        self._val_reproj_sum_r = defaultdict(float)

    def on_validation_epoch_end(self):
        """
        I8: Compute and log per-camera reprojection error for both
        base and refined extrinsics.  Uses Welford-style online
        accumulation (sum + count) from validation_step to avoid
        storing full tensors.
        """
        if not self.learn_calibration:
            return
        if self._val_reproj_n is None:
            return

        S = self.cal_refine.num_cameras

        print("\n" + "-" * 70)
        print("PER-CAMERA REPROJECTION ERROR (validation epoch)")
        print("-" * 70)
        print(f"{'Cam':<6} {'N':<8} {'Base(px)':<12} "
              f"{'Refined(px)':<14} {'Δ(px)':<10}")
        print("-" * 70)

        for c in range(S):
            n = self._val_reproj_n.get(c, 0)
            if n == 0:
                print(f"{c:<6} {'0':<8} {'N/A':<12} "
                      f"{'N/A':<14} {'N/A':<10}")
                continue
            base_mean = self._val_reproj_sum[c] / n
            ref_mean = self._val_reproj_sum_r[c] / n
            delta = base_mean - ref_mean

            print(f"{c:<6} {n:<8} {base_mean:<12.2f} "
                  f"{ref_mean:<14.2f} {delta:<+10.2f}")

            self.log(f'calib/cam{c}_reproj_err_base_px',
                     base_mean, on_epoch=True)
            self.log(f'calib/cam{c}_reproj_err_refined_px',
                     ref_mean, on_epoch=True)
            self.log(f'calib/cam{c}_reproj_improvement_px',
                     delta, on_epoch=True)

        print("-" * 70 + "\n")

    @torch.no_grad()
    def _accumulate_reproj_diagnostics(self, item):
        """
        I8: For each camera, compute pixel reprojection error using
        both base and refined extrinsics.  Accumulates running sums
        for online mean computation (no full tensors stored).
        """
        grid_gt = item['grid_gt']
        img_gt_2d = item['img_gt_2d']
        K_orig = item['intrinsic_original']
        E_base = item['extrinsic']
        E_refined = self.cal_refine(E_base)
        W = item['worldcoord_from_worldgrid']

        B = grid_gt.shape[0]
        S = img_gt_2d.shape[1]
        device = grid_gt.device

        if W.dim() == 2:
            W = W.unsqueeze(0).expand(B, -1, -1)

        for b in range(B):
            valid_mask = grid_gt[b, :, 2] > 0
            if not valid_mask.any():
                continue

            gc = grid_gt[b][valid_mask]
            gx = gc[:, 0].float()
            gy = gc[:, 1].float()
            pids = gc[:, 2].long()
            N = pids.shape[0]

            Wb = W[b]
            wx = Wb[0, 0] * gx + Wb[0, 1] * gy + Wb[0, 2]
            wy = Wb[1, 0] * gx + Wb[1, 1] * gy + Wb[1, 2]
            wh = torch.stack([
                wx, wy,
                torch.zeros(N, device=device, dtype=wx.dtype),
                torch.ones(N, device=device, dtype=wx.dtype),
            ], dim=1)

            for c in range(S):
                ig = img_gt_2d[b, c]
                dp = ig[:, 4].long()
                bv = ~((ig[:, 0] == -1) & (ig[:, 1] == -1) &
                       (ig[:, 2] == -1) & (ig[:, 3] == -1))
                dv = (dp > 0) & bv
                if not dv.any():
                    continue
                vd = ig[dv]
                vdp = dp[dv]
                fu = (vd[:, 0] + vd[:, 2]) / 2.0
                fv = vd[:, 3]

                for ext_key, E in [('base', E_base[b, c]),
                                   ('refined', E_refined[b, c])]:
                    K = K_orig[b, c]
                    cp = E[:3, :] @ wh.T
                    cz = cp[2, :].clamp(min=1e-4)
                    inf = cp[2, :] > 0
                    up = K[0, 0] * cp[0, :] / cz + K[0, 2]
                    vp = K[1, 1] * cp[1, :] / cz + K[1, 2]

                    cam_err_sum = 0.0
                    cam_n = 0
                    for i in range(N):
                        if not inf[i]:
                            continue
                        mi = (vdp == pids[i]).nonzero(
                            as_tuple=True
                        )[0]
                        if len(mi) == 0:
                            continue
                        j = mi[0]
                        err = (
                                (up[i] - fu[j]).pow(2)
                                + (vp[i] - fv[j]).pow(2)
                        ).sqrt().item()
                        cam_err_sum += err
                        cam_n += 1

                    if cam_n > 0:
                        if ext_key == 'base':
                            self._val_reproj_n[c] += cam_n
                            self._val_reproj_sum[c] += cam_err_sum
                        else:
                            self._val_reproj_sum_r[c] += cam_err_sum

    def on_train_epoch_end(self):
        """Save refined calibrations after every training epoch."""
        if not self.learn_calibration:
            return
        self.save_refined_calibrations(epoch=self.current_epoch)

    def save_refined_calibrations(self, epoch: int):
        """
        Export the current learned corrections to
            <trainer.log_dir>/calibrations/<CAM>_extrinsic.npz
        for human inspection and optional reloading.
        """
        if not self.learn_calibration:
            return
        if not (hasattr(self, 'trainer') and self.trainer is not None):
            return

        log_dir = (self.trainer.log_dir
                   if self.trainer.log_dir else '../data/cache')
        calib_dir = os.path.join(log_dir, 'calibrations')
        os.makedirs(calib_dir, exist_ok=True)

        base_dict, cam_names = (
            self.trainer.datamodule.get_base_extrinsics()
        )
        refined_dict = self.cal_refine.export_refined_extrinsics(
            base_dict, cam_names
        )

        # Determine dataset root for base_source heuristic
        dm = self.trainer.datamodule
        ds = (dm.data_train or dm.data_val
              or dm.data_test or dm.data_predict)
        seq_root = None
        if ds is not None:
            if hasattr(ds, 'datasets') and ds.datasets:
                seq_root = ds.datasets[0].base.root
            elif hasattr(ds, 'base'):
                seq_root = ds.base.root

        norms = self.cal_refine.get_delta_norms()
        delta_r_all = self.cal_refine.delta_r.detach().cpu().numpy()
        delta_t_all = self.cal_refine.delta_t.detach().cpu().numpy()

        print(f"\n{'Camera':<8} | {'‖Δr‖ (deg)':<12} | "
              f"{'‖Δt‖ (world)':<14} | saved to")
        print("-" * 70)

        for c, name in enumerate(cam_names):
            Rt = refined_dict[name]
            dr = delta_r_all[c]
            dt = delta_t_all[c]
            dr_norm = norms[c][0]
            dt_norm = norms[c][1]

            # Determine base_source
            base_source = "extrinsic_original"
            if seq_root is not None:
                recovered_path = os.path.join(
                    seq_root, 'calibrations', 'extrinsic_recovered',
                    f'{name}_extrinsic.npz',
                )
                if os.path.exists(recovered_path):
                    base_source = "extrinsic_recovered"

            out_path = os.path.join(
                calib_dir, f'{name}_extrinsic.npz'
            )
            np.savez(
                out_path,
                Rt=Rt,
                R=Rt[:, :3],
                t=Rt[:, 3],
                delta_r=dr,
                delta_t=dt,
                delta_r_norm=np.float32(dr_norm),
                delta_t_norm=np.float32(dt_norm),
                epoch=epoch,
                method="learned_refinement",
                base_source=base_source,
            )

            dr_deg = np.degrees(dr_norm)
            print(f"{name:<8} | {dr_deg:<12.4f} | "
                  f"{dt_norm:<14.4f} | {out_path}")


    # ==================================================================
    # Test
    # ==================================================================
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        hp = checkpoint.get('hyper_parameters', {})
        for key in (
            'use_2d_tracking',
            'tau_score',
            'tau_match',
            'original_img_hw',
            'save_2d_visualizations',
            'viz_every_n_frames',
            'reset_calibration_at_test',
            'calibration_dir',
            # ── NEW ──
            'use_foot_point',
            'adaptive_tau',
            'tau_floor',
            'tau_fraction',
            'cost_gate_factor',
            'projection_margin',
        ):
            hp.pop(key, None)

    def on_test_start(self):
        """
        Load, reset, or keep calibration refinement parameters at test time.

        Three modes:
          1. reset_calibration_at_test=True  → zero all deltas (ablation)
          2. calibration_dir is set          → load deltas from .npz files
          3. Neither                         → use deltas from checkpoint
        """
        print("\n" + "=" * 60)
        print("CALIBRATION REFINEMENT STATUS (on_test_start)")
        print("=" * 60)

        if not self.learn_calibration:
            print("  learn_calibration = False")
            print("  → Module not instantiated; using base extrinsics")
            print("=" * 60 + "\n")
            return

        print(f"  learn_calibration          = True")
        print(f"  reset_calibration_at_test  = "
              f"{self.reset_calibration_at_test}")
        print(f"  calibration_dir            = '{self.calibration_dir}'")

        # Show current values loaded from checkpoint
        print(f"\n  Deltas loaded from checkpoint:")
        for cam_idx, (dr, dt) in \
                self.cal_refine.get_delta_norms().items():
            print(f"    cam{cam_idx}: ‖Δr‖={dr:.6f} rad "
                  f"({np.degrees(dr):.4f}°), ‖Δt‖={dt:.6f}")

        # ── Mode 1: Reset to zero (ablation) ─────────────────
        if self.reset_calibration_at_test:
            print("\n  ★ RESET MODE: zeroing all deltas")
            with torch.no_grad():
                self.cal_refine.delta_r.zero_()
                self.cal_refine.delta_t.zero_()
            print("  ✓ All deltas set to zero.")
            print("  → Forward pass will use base extrinsics "
                  "unchanged.")
            print("=" * 60 + "\n")
            return

        # ── Mode 2: Load from directory ───────────────────────
        if self.calibration_dir:
            print(f"\n  ★ LOAD MODE: reading .npz files from "
                  f"calibration_dir")

            _, cam_names = (
                self.trainer.datamodule.get_base_extrinsics()
            )

            loaded = 0
            for c, name in enumerate(cam_names):
                # Try two path patterns:
                #   Pattern A: calibration_dir/calibrations/<cam>.npz
                #   Pattern B: calibration_dir/<cam>.npz
                path_a = os.path.join(
                    self.calibration_dir, 'calibrations',
                    f'{name}_extrinsic.npz',
                )
                path_b = os.path.join(
                    self.calibration_dir,
                    f'{name}_extrinsic.npz',
                )

                path = None
                if os.path.exists(path_a):
                    path = path_a
                elif os.path.exists(path_b):
                    path = path_b

                if path is None:
                    print(f"  ✗ {name}: NOT FOUND")
                    print(f"      tried: {path_a}")
                    print(f"      tried: {path_b}")
                    continue

                data = np.load(path, allow_pickle=True)

                if ('delta_r' not in data
                        or 'delta_t' not in data):
                    print(f"  ✗ {name}: found {path}")
                    print(f"      but MISSING delta_r/delta_t keys")
                    print(f"      available keys: "
                          f"{list(data.keys())}")
                    print(f"      This file is NOT a learned-"
                          f"refinement export. Skipping.")
                    continue

                with torch.no_grad():
                    self.cal_refine.delta_r[c] = torch.tensor(
                        data['delta_r'], dtype=torch.float32,
                        device=self.cal_refine.delta_r.device,
                    )
                    self.cal_refine.delta_t[c] = torch.tensor(
                        data['delta_t'], dtype=torch.float32,
                        device=self.cal_refine.delta_t.device,
                    )
                loaded += 1

                dr_norm = float(data['delta_r_norm'])
                dt_norm = float(data['delta_t_norm'])
                print(f"  ✓ {name}: loaded from {path}")
                print(f"      ‖Δr‖={dr_norm:.4f} rad "
                      f"({np.degrees(dr_norm):.4f}°), "
                      f"‖Δt‖={dt_norm:.4f}")

            print(f"\n  Loaded: {loaded}/{len(cam_names)} cameras")
            if loaded == 0:
                print("  ⚠ WARNING: No calibrations loaded!")
                print("    The checkpoint deltas will be used "
                      "unchanged.")
                print("    To test WITHOUT refinement, use:")
                print("      --model.reset_calibration_at_test true")

        # ── Mode 3: Use checkpoint (default) ──────────────────
        else:
            print("\n  ★ CHECKPOINT MODE: using deltas from "
                  "checkpoint as-is")

        # Final confirmation
        print(f"\n  Final deltas that WILL BE USED:")
        for cam_idx, (dr, dt) in \
                self.cal_refine.get_delta_norms().items():
            print(f"    cam{cam_idx}: ‖Δr‖={dr:.6f} rad "
                  f"({np.degrees(dr):.4f}°), ‖Δt‖={dt:.6f}")
        print("=" * 60 + "\n")

    def on_test_epoch_start(self):
        self._current_test_seq = -1
        self.test_tracker = JDETracker(conf_thres=self.conf_threshold)
        self.moda_gt_list, self.moda_pred_list = [], []
        self.mota_gt_list, self.mota_pred_list = [], []
        self.inference_times = []
        self.test_start_time = None
        self.cam2d_pred = defaultdict(list)
        self.cam2d_gt = defaultdict(list)
        self._2d_frame_counter = 0

        # ── 2D tracking init with LOUD diagnostics ──────── ◄◄◄
        print("\n" + "=" * 60)
        print("2D TRACKING SETUP")
        print("=" * 60)
        print(f"  use_2d_tracking = {self.use_2d_tracking}")
        print(f"  HAS_YOLO        = {HAS_YOLO}")
        print(f"  tau_score        = {self.tau_score}")
        print(f"  tau_match        = {self.tau_match}")
        print(f"  use_foot_point     = {self.use_foot_point}")
        print(f"  adaptive_tau       = {self.adaptive_tau}")
        print(f"  tau_floor          = {self.tau_floor}")
        print(f"  tau_fraction       = {self.tau_fraction}")
        print(f"  cost_gate_factor   = {self.cost_gate_factor}")
        print(f"  projection_margin  = {self.projection_margin}")
        print(f"  original_img_hw  = {self.original_img_hw}")
        print(f"  save_2d_viz      = {self.save_2d_visualizations}")

        if not self.use_2d_tracking:
            print("  → 2D tracking DISABLED by config")
            print("=" * 60 + "\n")
            return

        if not HAS_YOLO:
            print("  ✗ ultralytics NOT installed!")
            print("    Fix: pip install ultralytics")
            print("  → DISABLING 2D tracking")
            self.use_2d_tracking = False
            print("=" * 60 + "\n")
            return

        # Resolve YOLO weights path
        abs_yolo = self.yolo_weights_path
        if not os.path.isabs(abs_yolo):
            abs_yolo = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                abs_yolo,
            )
        print(f"  YOLO weights path: {abs_yolo}")
        print(f"  Exists: {os.path.exists(abs_yolo)}")

        if not os.path.exists(abs_yolo):
            print(f"  ✗ YOLO weights NOT FOUND at {abs_yolo}")
            print("  → DISABLING 2D tracking")
            self.use_2d_tracking = False
            print("=" * 60 + "\n")
            return

        try:
            self.yolo = _YOLO(abs_yolo)
            print(f"  ✓ YOLO loaded successfully")
            print(f"    Model type: {type(self.yolo)}")

            # Quick sanity: check model info
            if hasattr(self.yolo, 'names'):
                print(f"    Classes: {self.yolo.names}")

        except Exception as e:
            print(f"  ✗ YOLO load FAILED: {e}")
            print("  → DISABLING 2D tracking")
            self.use_2d_tracking = False
            self.yolo = None

        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        item, target = batch

        start_time = time.perf_counter()
        output = self(item)

        center_e = output['instance_center']
        offset_e = output['instance_offset']
        xy_e, xy_prev_e, scores_e, classes_e = decode.decoder(
            center_e.sigmoid(), offset_e, None,
            K=self.max_detections,
        )
        mem_xyz = torch.cat(
            (xy_e, torch.zeros_like(xy_e[..., 0:1])), dim=2
        )
        ref_xy = self.vox_util.Mem2Ref(
            mem_xyz, self.Y, self.Z, self.X
        )[..., :2]
        mem_xyz_prev = torch.cat(
            (xy_prev_e, torch.zeros_like(xy_e[..., 0:1])), dim=2
        )
        ref_xy_prev = self.vox_util.Mem2Ref(
            mem_xyz_prev, self.Y, self.Z, self.X
        )[..., :2]

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        batch_size = item['frame'].shape[0]
        time_per_frame = inference_time_ms / batch_size
        for frame, seq_num in zip(
                item['frame'], item['sequence_num']):
            self.inference_times.append([
                int(seq_num.item()), int(frame.item()),
                time_per_frame,
            ])

        # ── detection metrics ─────────────────────────────────
        for frame, grid_gt, xy, score, seq_num in zip(
            item['frame'], item['grid_gt'], ref_xy, scores_e,
            item['sequence_num'],
        ):
            frame_val = int(frame.item())
            seq_val = int(seq_num.item())
            global_frame = seq_val * 1_000_000 + frame_val
            valid = score > self.conf_threshold
            self.moda_gt_list.extend([
                [global_frame, x.item(), y.item()]
                for x, y, _ in grid_gt[grid_gt.sum(1) != 0]
            ])
            self.moda_pred_list.extend([
                [global_frame, x.item(), y.item()]
                for x, y in xy[valid]
            ])

        # ── BEV tracking + 2D backward matching ──────────────
        for b_idx, (seq_num, frame, grid_gt, bev_det, bev_prev,
                     score) in enumerate(zip(
            item['sequence_num'], item['frame'], item['grid_gt'],
            ref_xy.cpu(), ref_xy_prev.cpu(), scores_e.cpu(),
        )):
            frame_val = int(frame.item())
            seq_val = int(seq_num.item())

            if seq_val != self._current_test_seq:
                self.test_tracker = JDETracker(
                    conf_thres=self.conf_threshold
                )
                self._current_test_seq = seq_val

            output_stracks = self.test_tracker.update(
                bev_det, bev_prev, score
            )

            # BEV MOT accumulation
            self.mota_gt_list.extend([
                [seq_val, frame_val, i.item(),
                 -1, -1, -1, -1, 1,
                 x.item(), y.item(), -1]
                for x, y, i in grid_gt[grid_gt.sum(1) != 0]
            ])
            self.mota_pred_list.extend([
                [seq_val, frame_val, s.track_id,
                 -1, -1, -1, -1, float(s.score)]
                + s.xy.tolist() + [-1]
                for s in output_stracks
            ])

            # ── 2D backward matching ─────────────────── ◄◄◄
            if self.use_2d_tracking and self.yolo is not None:
                self._2d_frame_counter += 1
                try:
                    self._run_2d_tracking(
                        item, b_idx, output_stracks,
                        seq_val, frame_val,
                        output=output,
                    )
                except Exception as e:
                    if self._2d_frame_counter <= 3:
                        print(f"\n  [2D-TRACK] ERROR on frame "
                              f"{frame_val}: {e}")
                        import traceback
                        traceback.print_exc()

    # ------------------------------------------------------------------
    def _run_2d_tracking(self, item, b_idx, output_stracks,
                         seq_val, frame_val, output=None):
        """Run YOLO + backward matching for one frame."""
        num_cams = item['img'].shape[1]
        verbose = (self._2d_frame_counter <= 3)

        if verbose:
            print(f"\n{'='*60}")
            print(f"[2D-TRACK] seq={seq_val} frame={frame_val} "
                  f"num_stracks={len(output_stracks)}")

        # ── Unpack image paths ────────────────────────────
        paths_raw = item['img_paths']
        if isinstance(paths_raw, (list, tuple)):
            paths_str = paths_raw[b_idx]
        else:
            paths_str = paths_raw
        cam_paths = paths_str.split('||')

        if verbose:
            print(f"  img_paths type: {type(paths_raw)}")
            for i, p in enumerate(cam_paths):
                print(f"    cam {i}: {p} "
                      f"exists={os.path.exists(p)}")

        # ── Homographies (use refined extrinsics if available) ──
        intrinsic_orig = item['intrinsic_original'][b_idx]
        if (self.learn_calibration
                and output is not None
                and 'extrinsic_refined' in output):
            extrinsic_for_homo = output['extrinsic_refined'][b_idx]
        else:
            extrinsic_for_homo = item['extrinsic'][b_idx]

        ref_T_global = item['ref_T_global'][b_idx]

        homographies = compute_ground_plane_homographies(
            intrinsic_orig, extrinsic_for_homo, ref_T_global,
            verbose=verbose,
        )

        # ── BEV track positions / ids ─────────────────────
        bev_positions = [s.xy.tolist() for s in output_stracks]
        bev_ids = [int(s.track_id) for s in output_stracks]

        if verbose:
            print(f"\n  BEV tracks: {len(bev_positions)}")
            for i, (pos, tid) in enumerate(
                zip(bev_positions[:5], bev_ids[:5])
            ):
                print(f"    Track {tid}: grid=({pos[0]:.1f},"
                      f"{pos[1]:.1f})")

        # ── Run YOLO on each camera ───────────────────────
        detections_per_cam = {}
        for cam_idx in range(num_cams):
            if cam_idx >= len(cam_paths):
                if verbose:
                    print(f"\n  Cam {cam_idx}: NO PATH")
                detections_per_cam[cam_idx] = []
                continue
            path = cam_paths[cam_idx]
            if not os.path.exists(path):
                if verbose:
                    print(f"\n  Cam {cam_idx}: FILE NOT FOUND: "
                          f"{path}")
                detections_per_cam[cam_idx] = []
                continue

            try:
                yolo_results = self.yolo(
                    path, verbose=False,
                    conf=self.tau_score * 0.5,
                )
            except Exception as e:
                if verbose:
                    print(f"\n  Cam {cam_idx}: YOLO FAILED: {e}")
                detections_per_cam[cam_idx] = []
                continue

            dets = []
            for r in yolo_results:
                boxes = r.boxes
                if boxes is not None and len(boxes):
                    for box, conf in zip(
                        boxes.xyxy.cpu().numpy(),
                        boxes.conf.cpu().numpy(),
                    ):
                        dets.append((
                            float(box[0]), float(box[1]),
                            float(box[2]), float(box[3]),
                            float(conf),
                        ))
            detections_per_cam[cam_idx] = dets

            if verbose:
                print(f"\n  Cam {cam_idx}: YOLO found "
                      f"{len(dets)} detections")
                for d in dets[:3]:
                    print(f"    bbox=({d[0]:.0f},{d[1]:.0f},"
                          f"{d[2]:.0f},{d[3]:.0f}) "
                          f"conf={d[4]:.2f}")

        # ── Backward matching (P1–P4) ─────────────────────
        cam_results = assign_ids_to_detections(
            bev_positions, bev_ids,
            detections_per_cam, homographies,
            img_hw=self.original_img_hw,
            tau_score=self.tau_score,
            tau_match=self.tau_match,
            use_foot_point=self.use_foot_point,
            adaptive_tau=self.adaptive_tau,
            tau_floor=self.tau_floor,
            tau_fraction=self.tau_fraction,
            cost_gate_factor=self.cost_gate_factor,
            projection_margin=self.projection_margin,
            verbose=verbose,
        )

        # ── Summary of this frame ─────────────────────────
        total_matched = 0
        total_unmatched = 0
        for cam_idx, res in cam_results.items():
            matched = sum(1 for r in res if r[5] >= 0)
            unmatched = sum(1 for r in res if r[5] < 0)
            total_matched += matched
            total_unmatched += unmatched

        if verbose:
            print(f"\n [2D-TRACK] SUMMARY: "
                  f"{total_matched} matched, "
                  f"{total_unmatched} unmatched across "
                  f"{num_cams} cameras")
            print(f"{'='*60}\n")

        # ── Visualization ─────────────────────────────────
        if (self.save_2d_visualizations and
                self._2d_frame_counter % self.viz_every_n_frames == 0):
            log_dir = (
                self.trainer.log_dir
                if self.trainer.log_dir is not None
                else '../data/cache'
            )
            viz_dir = os.path.join(log_dir, '2d_visualizations')
            visualize_2d_tracking(
                cam_paths=cam_paths,
                cam_results=cam_results,
                homographies=homographies,
                bev_positions=bev_positions,
                bev_ids=bev_ids,
                save_dir=viz_dir,
                frame_id=frame_val,
                seq_id=seq_val,
                img_hw=self.original_img_hw,
                use_foot_point=self.use_foot_point,
            )

        # ── Log match rate per camera for monitoring ──────
        self.log('match_rate/total',
                 total_matched / max(total_matched + total_unmatched, 1),
                 on_step=True, on_epoch=True, prog_bar=False)
        for cam_idx, res in cam_results.items():
            matched = sum(1 for r in res if r[5] >= 0)
            total = len(res)
            self.log(f'match_rate/cam{cam_idx}',
                     matched / max(total, 1),
                     on_step=False, on_epoch=True)

        # ── Accumulate predictions ────────────────────────
        for cam_idx, results_list in cam_results.items():
            key = (seq_val, cam_idx)
            for (x1, y1, x2, y2, s, tid) in results_list:
                if tid < 0:
                    continue
                w = x2 - x1
                h = y2 - y1
                self.cam2d_pred[key].append([
                    frame_val, tid, x1, y1, w, h, s,
                ])

        # ── Accumulate ground truth ───────────────────────
        img_gt_2d = item['img_gt_2d'][b_idx]
        for cam_idx in range(num_cams):
            key = (seq_val, cam_idx)
            gt_cam = img_gt_2d[cam_idx]
            for obj in gt_cam:
                if obj.sum().item() == 0:
                    continue
                x1, y1, x2, y2, pid = (
                    obj[0].item(), obj[1].item(),
                    obj[2].item(), obj[3].item(),
                    int(obj[4].item()),
                )
                if pid <= 0:
                    continue
                w = x2 - x1
                h = y2 - y1
                self.cam2d_gt[key].append([
                    frame_val, pid, x1, y1, w, h,
                ])

    # ------------------------------------------------------------------
    def on_test_epoch_end(self):
        import pandas as pd

        log_dir = (
            self.trainer.log_dir
            if self.trainer.log_dir is not None
            else '../data/cache'
        )
        os.makedirs(log_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        # ══════════════════════════════════════════════════
        # 1. OVERALL DETECTION METRICS
        # ══════════════════════════════════════════════════
        pred_path = osp.join(log_dir, 'moda_pred.txt')
        gt_path = osp.join(log_dir, 'moda_gt.txt')
        np.savetxt(pred_path, np.array(self.moda_pred_list), '%f')
        np.savetxt(gt_path, np.array(self.moda_gt_list), '%d')
        recall, precision, moda, modp = modMetricsCalculator(
            osp.abspath(pred_path), osp.abspath(gt_path)
        )
        print("\n" + "-" * 80)
        print("OVERALL DETECTION METRICS")
        print("-" * 80)
        print(f"  Recall:     {recall:.2f}%")
        print(f"  Precision:  {precision:.2f}%")
        print(f"  MODA:       {moda:.2f}%")
        print(f"  MODP:       {modp:.2f}%")
        self.log('detect/overall_recall', recall)
        self.log('detect/overall_precision', precision)
        self.log('detect/overall_moda', moda)
        self.log('detect/overall_modp', modp)

        # ══════════════════════════════════════════════════
        # 2. PER-SEQUENCE DETECTION METRICS
        # ══════════════════════════════════════════════════
        if self.moda_gt_list:
            gt_arr = np.array(self.moda_gt_list)
            pred_arr = (
                np.array(self.moda_pred_list)
                if self.moda_pred_list else np.empty((0, 3))
            )
            unique_seqs = sorted(np.unique(
                (gt_arr[:, 0] // 1_000_000).astype(int)
            ))
            print("\n" + "-" * 80)
            print("PER-SEQUENCE DETECTION METRICS")
            print("-" * 80)
            print(f"{'Seq':<6} {'Recall':<10} {'Precision':<10} "
                  f"{'MODA':<10} {'MODP':<10}")
            print("-" * 80)
            seq_detection_results = []
            for seq in unique_seqs:
                mask_gt = (
                    (gt_arr[:, 0] // 1_000_000).astype(int) == seq
                )
                mask_pred = (
                    (pred_arr[:, 0] // 1_000_000).astype(int) == seq
                    if len(pred_arr) > 0
                    else np.zeros(0, dtype=bool)
                )
                seq_gt_path = osp.join(
                    log_dir, f'moda_gt_seq{int(seq)}.txt'
                )
                seq_pred_path = osp.join(
                    log_dir, f'moda_pred_seq{int(seq)}.txt'
                )
                np.savetxt(seq_gt_path, gt_arr[mask_gt], '%f')
                if mask_pred.any():
                    np.savetxt(
                        seq_pred_path, pred_arr[mask_pred], '%f'
                    )
                else:
                    np.savetxt(
                        seq_pred_path, np.empty((0, 3)), '%f'
                    )
                if mask_pred.any():
                    sr, sp, sm, smp = modMetricsCalculator(
                        osp.abspath(seq_pred_path),
                        osp.abspath(seq_gt_path),
                    )
                else:
                    sr, sp, sm, smp = 0., 0., 0., 0.
                print(f"{seq:<6} {sr:<10.2f} {sp:<10.2f} "
                      f"{sm:<10.2f} {smp:<10.2f}")
                self.log(f'detect/seq{seq}_recall', sr)
                self.log(f'detect/seq{seq}_precision', sp)
                self.log(f'detect/seq{seq}_moda', sm)
                self.log(f'detect/seq{seq}_modp', smp)
                seq_detection_results.append({
                    'sequence': seq, 'recall': sr,
                    'precision': sp, 'moda': sm, 'modp': smp,
                })
            det_df = pd.DataFrame(seq_detection_results)
            det_csv = osp.join(
                log_dir, 'detection_metrics_per_sequence.csv'
            )
            det_df.to_csv(det_csv, index=False)
            print(f"\nSaved to: {det_csv}")

        # ══════════════════════════════════════════════════
        # 3. OVERALL TRACKING METRICS
        # ══════════════════════════════════════════════════
        scale = 1 if self.X == 150 else 0.025
        pred_path = osp.join(log_dir, 'mota_pred.txt')
        gt_path = osp.join(log_dir, 'mota_gt.txt')
        np.savetxt(
            pred_path, np.array(self.mota_pred_list),
            '%f', delimiter=',',
        )
        np.savetxt(
            gt_path, np.array(self.mota_gt_list),
            '%f', delimiter=',',
        )
        summary = mot_metrics(
            osp.abspath(pred_path), osp.abspath(gt_path), scale
        )
        summary = summary.loc['OVERALL']
        print("\n" + "-" * 80)
        print("OVERALL TRACKING METRICS")
        print("-" * 80)
        print(f"  MOTA:  {summary['mota']:.2f}%")
        print(f"  MOTP:  {100 - summary['motp']:.2f}%")
        print(f"  IDF1:  {summary['idf1']:.2f}%")
        print(f"  MT:    {summary['mostly_tracked']:.2f}%")
        print(f"  ML:    {summary['mostly_lost']:.2f}%")
        for key, value in summary.to_dict().items():
            if value >= 1 and key[:3] != 'num':
                value /= summary.to_dict()['num_unique_objects']
            value = value * 100 if value < 1 else value
            value = 100 - value if key == 'motp' else value
            self.log(f'track/{key}', value)

        # ── 4. Per-sequence BEV tracking debug files ──────
        if self.mota_gt_list:
            gt_arr = np.array(self.mota_gt_list)
            pred_arr = (
                np.array(self.mota_pred_list)
                if self.mota_pred_list else np.empty((0, 11))
            )
            for seq in np.unique(gt_arr[:, 0].astype(int)):
                seq_gt = gt_arr[gt_arr[:, 0].astype(int) == seq]
                seq_pred = (
                    pred_arr[pred_arr[:, 0].astype(int) == seq]
                    if len(pred_arr) else np.empty((0, 11))
                )
                np.savetxt(
                    osp.join(
                        log_dir, f'mota_gt_seq{int(seq)}.txt'
                    ),
                    seq_gt, '%f', delimiter=','
                )
                np.savetxt(
                    osp.join(
                        log_dir, f'mota_pred_seq{int(seq)}.txt'
                    ),
                    seq_pred, '%f', delimiter=','
                )

        # ══════════════════════════════════════════════════
        # 5. PER-CAMERA 2D TRACKING METRICS               ◄◄◄
        # ══════════════════════════════════════════════════
        print("\n" + "-" * 80)
        print("2D PER-CAMERA TRACKING STATUS")
        print("-" * 80)
        print(f"  use_2d_tracking:  {self.use_2d_tracking}")
        print(f"  YOLO loaded:      {self.yolo is not None}")
        print(f"  Frames processed: {self._2d_frame_counter}")
        print(f"  cam2d_pred keys:  {len(self.cam2d_pred)}")
        print(f"  cam2d_gt keys:    {len(self.cam2d_gt)}")

        total_pred = sum(len(v) for v in self.cam2d_pred.values())
        total_gt = sum(len(v) for v in self.cam2d_gt.values())
        print(f"  Total 2D preds:   {total_pred}")
        print(f"  Total 2D gt:      {total_gt}")

        # Always report, even if empty
        if self.use_2d_tracking:
            if not self.cam2d_pred and not self.cam2d_gt:
                print("\n  ✗ NO 2D tracking data collected!")
                print("    Possible causes:")
                print("    1. YOLO produced no detections")
                print("    2. All matches exceeded tau_match")
                print("    3. img_paths not resolved correctly")
                print("    Check verbose output from first "
                      "3 frames above.\n")
            else:
                all_keys = sorted(
                    set(self.cam2d_gt.keys())
                    | set(self.cam2d_pred.keys())
                )
                cam2d_results = []

                print(f"\n  {'Seq':>4} {'Cam':>4} {'MOTA':>8} "
                      f"{'IDF1':>8} {'IDSW':>6} "
                      f"{'GT':>6} {'Pred':>6}")
                print("  " + "-" * 50)

                for key in all_keys:
                    seq_val, cam_idx = key
                    gt_rows = self.cam2d_gt.get(key, [])
                    pr_rows = self.cam2d_pred.get(key, [])

                    # Save MOT-format files
                    mot_dir = osp.join(
                        log_dir, '2d_tracking',
                        f'seq{seq_val}', f'cam{cam_idx}'
                    )
                    os.makedirs(mot_dir, exist_ok=True)

                    gt_file = osp.join(mot_dir, 'gt.txt')
                    pr_file = osp.join(mot_dir, 'pred.txt')

                    if gt_rows:
                        np.savetxt(
                            gt_file, np.array(gt_rows), '%.2f',
                            delimiter=','
                        )
                    else:
                        np.savetxt(
                            gt_file, np.empty((0, 6)), '%.2f',
                            delimiter=','
                        )
                    if pr_rows:
                        np.savetxt(
                            pr_file, np.array(pr_rows), '%.2f',
                            delimiter=','
                        )
                    else:
                        np.savetxt(
                            pr_file, np.empty((0, 7)), '%.2f',
                            delimiter=','
                        )

                    # Evaluate
                    summary_2d = evaluate_2d_mot(
                        gt_rows, pr_rows, iou_threshold=0.5
                    )

                    if summary_2d is not None:
                        mota_2d = summary_2d.get('mota', 0.0)
                        idf1_2d = summary_2d.get('idf1', 0.0)
                        idsw_2d = summary_2d.get(
                            'num_switches', 0
                        )
                    else:
                        mota_2d = idf1_2d = 0.0
                        idsw_2d = 0

                    mota_pct = (mota_2d * 100
                                if abs(mota_2d) < 2
                                else mota_2d)
                    idf1_pct = (idf1_2d * 100
                                if abs(idf1_2d) < 2
                                else idf1_2d)

                    print(f"  {seq_val:>4} {cam_idx:>4} "
                          f"{mota_pct:>8.1f} {idf1_pct:>8.1f} "
                          f"{idsw_2d:>6} "
                          f"{len(gt_rows):>6} {len(pr_rows):>6}")

                    self.log(
                        f'track2d/seq{seq_val}_cam{cam_idx}_mota',
                        mota_pct,
                    )
                    self.log(
                        f'track2d/seq{seq_val}_cam{cam_idx}_idf1',
                        idf1_pct,
                    )
                    cam2d_results.append({
                        'sequence': seq_val, 'camera': cam_idx,
                        'mota': mota_pct, 'idf1': idf1_pct,
                        'idsw': idsw_2d,
                        'num_gt': len(gt_rows),
                        'num_pred': len(pr_rows),
                    })

                if cam2d_results:
                    cam2d_df = pd.DataFrame(cam2d_results)
                    cam2d_csv = osp.join(
                        log_dir, 'tracking_2d_per_cam.csv'
                    )
                    cam2d_df.to_csv(cam2d_csv, index=False)
                    print(f"\n  Saved to: {cam2d_csv}")

        # ══════════════════════════════════════════════════
        # 6. INFERENCE TIMING
        # ══════════════════════════════════════════════════
        if self.inference_times:
            timing_arr = np.array(self.inference_times)
            timing_df = pd.DataFrame(
                timing_arr,
                columns=['sequence', 'frame', 'time_ms'],
            )
            overall_mean = timing_df['time_ms'].mean()
            overall_median = timing_df['time_ms'].median()
            overall_std = timing_df['time_ms'].std()
            print("\n" + "-" * 80)
            print("OVERALL INFERENCE TIMING (per image)")
            print("-" * 80)
            print(f"  Mean:   {overall_mean:.2f} ms")
            print(f"  Median: {overall_median:.2f} ms")
            print(f"  Std:    {overall_std:.2f} ms")
            print(f"  FPS:    {1000.0 / overall_mean:.2f}")
            self.log('timing/overall_mean_ms', overall_mean)
            self.log('timing/overall_median_ms', overall_median)
            self.log('timing/overall_fps', 1000.0 / overall_mean)

            print("\n" + "-" * 80)
            print("PER-SEQUENCE INFERENCE TIMING (per image)")
            print("-" * 80)
            print(f"{'Seq':<6} {'Frames':<8} {'Mean(ms)':<10} "
                  f"{'Median(ms)':<12} {'Std(ms)':<10} {'FPS':<8}")
            print("-" * 80)
            seq_timing = []
            for seq in sorted(timing_df['sequence'].unique()):
                sd = timing_df[timing_df['sequence'] == seq]
                sm = sd['time_ms'].mean()
                smd = sd['time_ms'].median()
                ss = sd['time_ms'].std()
                sf = 1000.0 / sm
                nf = len(sd)
                print(f"{int(seq):<6} {nf:<8} {sm:<10.2f} "
                      f"{smd:<12.2f} {ss:<10.2f} {sf:<8.2f}")
                self.log(f'timing/seq{int(seq)}_mean_ms', sm)
                self.log(f'timing/seq{int(seq)}_fps', sf)
                seq_timing.append({
                    'sequence': int(seq), 'num_frames': nf,
                    'mean_ms': sm, 'median_ms': smd,
                    'std_ms': ss, 'fps': sf,
                })
            pd.DataFrame(seq_timing).to_csv(
                osp.join(log_dir,
                         'inference_timing_per_sequence.csv'),
                index=False,
            )
            timing_df.to_csv(
                osp.join(log_dir,
                         'inference_timing_detailed.csv'),
                index=False,
            )

        # ══════════════════════════════════════════════════
        # 7. SAVE REFINED CALIBRATIONS (test snapshot)
        # ══════════════════════════════════════════════════
        if self.learn_calibration:
            print("\n" + "-" * 80)
            print("REFINED CALIBRATION EXPORT (test-time snapshot)")
            print("-" * 80)
            self.save_refined_calibrations(epoch=-1)

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80 + "\n")

    # ==================================================================
    # Plotting
    # ==================================================================
    def plot_data(self, target, output, batch_idx=0):
        center_e = output['instance_center']
        center_g = target['center_bev']
        writer = self.logger.experiment
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.imshow(
            center_g[-1].amax(0).sigmoid().squeeze().float().cpu().numpy()
        )
        ax2.imshow(
            center_e[-1].amax(0).sigmoid().squeeze().float().cpu().numpy()
        )
        ax1.set_title('center_g')
        ax2.set_title('center_e')
        plt.tight_layout()
        writer.add_figure(
            f'plot/{batch_idx}', fig, global_step=self.global_step
        )
        plt.close(fig)

    # ==================================================================
    # Optimizer
    # ==================================================================
    def configure_optimizers(self):
        if self.learn_calibration:
            calib_params = list(self.cal_refine.parameters())
            calib_param_ids = {id(p) for p in calib_params}
            other_params = [
                p for p in self.parameters()
                if id(p) not in calib_param_ids
            ]
            param_groups = [
                {'params': other_params,
                 'lr': self.learning_rate},
                {'params': calib_params,
                 'lr': self.learning_rate * self.lr_calib_scale},
            ]
            max_lrs = [
                self.learning_rate,
                self.learning_rate * self.lr_calib_scale,
            ]
        else:
            param_groups = self.parameters()
            max_lrs = self.learning_rate

        optimizer = torch.optim.Adam(param_groups)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_train_batch_start(self, batch, batch_idx):
        """
        I6: Calibration warmup schedule.
        Freezes calibration parameters until a warmup condition is met,
        then unfreezes them for the remainder of training.

        Two modes:
            'steps'          — unfreeze after calib_warmup_steps global steps
            'loss_threshold' — unfreeze when center_loss drops below threshold
        """
        if not self.learn_calibration:
            return
        if self._calib_unfrozen:
            return  # Already unfrozen; no-op

        should_unfreeze = False

        if self.calib_warmup_mode == 'steps':
            if self.global_step >= self.calib_warmup_steps:
                should_unfreeze = True
        elif self.calib_warmup_mode == 'loss_threshold':
            if self._latest_center_loss < self.calib_warmup_loss_threshold:
                should_unfreeze = True
        else:
            raise ValueError(
                f"Unknown calib_warmup_mode: {self.calib_warmup_mode}"
            )

        if should_unfreeze:
            for p in self.cal_refine.parameters():
                p.requires_grad = True
            self._calib_unfrozen = True

            # ── Reset the calib param-group LR to the start of its ramp ──
            optimizer = self.optimizers()
            if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 1:
                target_lr = self.learning_rate * self.lr_calib_scale
                optimizer.param_groups[1]['lr'] = target_lr
                # Also reset Adam state for a clean start
                for p in self.cal_refine.parameters():
                    if p in optimizer.state:
                        del optimizer.state[p]

            print(
                f"\n[WARMUP] Step {self.global_step}: "
                f"Calibration parameters UNFROZEN "
                f"(mode={self.calib_warmup_mode}, "
                f"center_loss={self._latest_center_loss:.4f})\n"
            )

# ======================================================================
if __name__ == '__main__':
    from lightning.pytorch.cli import LightningCLI

    torch.set_float32_matmul_precision('medium')

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments(
                "model.resolution", "data.init_args.resolution"
            )
            parser.link_arguments(
                "model.bounds", "data.init_args.bounds"
            )
            parser.link_arguments(
                "trainer.accumulate_grad_batches",
                "data.init_args.accumulate_grad_batches",
            )

    cli = MyLightningCLI(WorldTrackModel)