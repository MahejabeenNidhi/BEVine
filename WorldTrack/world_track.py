# WorldTrack/world_track.py
import os
import os.path as osp
import time
from collections import defaultdict

import torch
import torch.nn as nn
import lightning as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as _MplPolygon
import numpy as np

from models import Segnet, MVDet, Liftnet, Bevformernet
from models.loss import FocalLoss, reprojection_loss, reprojection_loss_3d
from models.calibration_refinement import CalibrationRefinementModule
from tracking.multitracker import JDETracker
from utils import vox, basic, decode, obb
from evaluation.mod import modMetricsCalculator
from evaluation.mot_bev import mot_metrics

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
        # Swin-T backbone toggle
        # "" (default)  -> BASELINE torchvision Swin-T (ImageNet).
        # non-empty     -> your lightly_train-pretrained Swin-T
        #                  (.pt export or .ckpt Lightning checkpoint).
        # Only used when encoder_name == 'swin_t'; ignored otherwise.
        swin_pretrained_path: str = "",
        swin_freeze_backbone: bool = False,
        img_grad_checkpoint: bool = False,
        # BEV duplicate suppression (decode-time NMS)
        # nms_kernel     : local-max window on the centre heat-map,
        #                  in BEV cells (odd, 3 = legacy).
        # nms_radius_cm  : hard centre-distance gate, in CENTIMETRES.
        # nms_iou        : rotated-BEV-IoU gate (needs the 3D heads).
        nms_kernel: int = 3,
        nms_radius_cm: float = 0.0,
        nms_iou: float = 0.0,
        # suppression must NOT be a function of the reporting threshold
        nms_score_threshold: float = 0.05,
        use_temporal_cache=True,
        z_sign=1,
        feat2d_dim=128,
        # image auxiliary (training)
        use_image_aux_loss: bool = False,
        img_aux_weight: float = 1.0,
        # calibration refinement
        learn_calibration: bool = False,
        cal_refine_reg_weight: float = 1e-4,
        lr_calib_scale: float = 0.1,
        calibration_dir: str = "",
        reset_calibration_at_test: bool = False,
        # reprojection loss
        use_reproj_loss: bool = True,
        reproj_weight: float = 0.1,
        min_obs_per_camera: int = 0,
        reproj_soft_gate: bool = False,
        reproj_soft_gate_temp: float = 1.0,
        reproj_loss_type: str = 'smooth_l1',
        reproj_robust_scale: float = 50.0,
        reproj_camera_weighting: str = 'proportional',
        # 3D-box corner reprojection loss
            use_reproj3d_loss: bool = False,
            reproj3d_weight: float = 0.1,
            reproj3d_loss_type: str = 'smooth_l1',
            reproj3d_normalize: bool = True,
            # If False the extrinsics are DETACHED inside the 3D loss, so a
            # badly-sized predicted box can never "fix itself" by moving the
            # cameras (which silently destroys the BEV<->image alignment).
            reproj3d_update_calib: bool = False,
            reproj3d_warmup_steps: float = 4000,
            # gradient attenuation for the 3D heads into the shared BEV trunk
            bev3d_grad_scale: float = 1.0,
            # per-object 3D attribute head (query-style)
            attr3d_neck_dim: int = 128,
            attr3d_head_dim: int = 128,
            attr3d_points: int = 8,
            attr3d_max_offset_cells: float = 20.0,
            attr3d_detach_trunk: bool = True,
            size_scale_cm: float = 100.0,
            # auxiliary 3D-loss containment
            aux3d_weight: float = 0.5,
            aux3d_warmup_steps: float = 0.25,
            aux3d_logvar_min: float = -1.0,
            aux3d_logvar_max: float = 3.0,
            yaw_mag_weight: float = 0.1,
            size_prior_weight: float = 0.01,
        # 3D evaluation
        eval_match_dist_cm: float = 50.0,
        # inference on unlabelled sequences (test time)
        unlabeled_use_3d: bool = True,
        # OBB evaluation (rotated-IoU CLEAR-MOD / CLEAR-MOT)
        obb_iou_thresholds: tuple = (0.10, 0.25, 0.50),
        # Rotated-IoU NMS applied to the OBB prediction stream only.
        obb_nms_iou: float = 0.5,
        # validation-time MODA (checkpoint selection metric)
        val_moda_interval: int = 5,
        val_conf_sweep: bool = True,
        val_moda_gate_cells: float = 40.0,
        # tracker gates expressed in CENTIMETRES, not Wildtrack cells
        track_assoc_dist_cm: float = 200.0,
        track_dup_dist_cm: float = 60.0,
        track_buffer: int = 5,
        # A lying cow must report ZERO motion: hold its Kalman velocity
        # at zero while the fused posture says "lying". Requires the 3D
        # attribute stream (posture).
        track_lying_velocity_freeze: bool = False,
        # 3D-AWARE TRACKING (test time)
        # False -> legacy behaviour: every track's OBB is re-sampled
        #          from the dense heads at that frame, no memory.
        # True  -> per-track running estimate of yaw / size / posture.
        track_use_3d_attrs: bool = False,
        track_size_mode: str = 'mean',  # 'mean' | 'ema' | 'median'
        track_size_alpha: float = 0.3,  # only for 'ema'
        track_size_median_window: int = 15,  # only for 'median'
        track_size_reject_cm: float = 0.0,  # 0 = accept every reading
        track_yaw_mode: str = 'ema',  # 'ema' | 'kf' (theta, omega)
        track_yaw_alpha: float = 0.3,
        track_yaw_flip_align: bool = True,  # handle the 180 deg ambiguity
        track_yaw_q: float = 4e-2,  # KF process noise
        track_yaw_r: float = 0.25,  # KF measurement noise (rad^2)
        track_posture_alpha: float = 0.3,
        track_posture_hysteresis: float = 0.1,
        # writeback of smoothed attributes to detection stream
        track_writeback_to_detections: bool = False,
        # TRACK-CONSISTENCY LOSS (train time)
        # Pull the size/heading predicted for a cow towards that same
        # cow's own recent (detached) history.
        use_track_attr_consistency: bool = False,
        track_attr_consistency_weight: float = 0.05,  # size
        track_yaw_consistency_weight: float = 0.0,  # heading (opt-in)
        # Rotated-IoU term in the association cost (0 = distance-only,
        # the legacy behaviour). Requires track_use_3d_attrs=True.
        track_assoc_use_iou: bool = False,
        track_assoc_iou_weight: float = 0.25,
        track_attr_bank_alpha: float = 0.3,
        track_attr_bank_max_gap: int = 3,  # frames
        track_attr_consistency_warmup_steps: float = 0.3,
        calib_warmup_steps: float = 0,
        calib_warmup_mode: str = 'steps',
        calib_warmup_loss_threshold: float = 1.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.learning_rate = learning_rate
        self.swin_pretrained_path = swin_pretrained_path or None
        self.swin_freeze_backbone = swin_freeze_backbone
        self.img_grad_checkpoint = bool(img_grad_checkpoint)
        self.resolution = resolution
        self.Y, self.Z, self.X = self.resolution
        self.bounds = bounds
        self.max_detections = max_detections
        self.D, self.DMIN, self.DMAX = depth
        self.conf_threshold = conf_threshold
        self.nms_kernel = int(nms_kernel)
        self.nms_radius_cm = float(nms_radius_cm)
        self.nms_iou = float(nms_iou)
        self._bev_cell_cm = None

        # Config
        self.use_image_aux_loss = use_image_aux_loss
        self.img_aux_weight = img_aux_weight

        # Calibration refinement
        self.learn_calibration = learn_calibration
        self.cal_refine_reg_weight = cal_refine_reg_weight
        self.lr_calib_scale = lr_calib_scale
        self.calibration_dir = calibration_dir
        self.reset_calibration_at_test = reset_calibration_at_test
        self.use_reproj_loss = use_reproj_loss
        self.reproj_weight = reproj_weight
        # minimum observation gate
        self.min_obs_per_camera = min_obs_per_camera
        self.reproj_soft_gate = reproj_soft_gate
        self.reproj_soft_gate_temp = reproj_soft_gate_temp
        self.reproj_loss_type = reproj_loss_type
        self.reproj_robust_scale = reproj_robust_scale
        self.reproj_camera_weighting = reproj_camera_weighting
        # 3D-box corner reprojection
        self.use_reproj3d_loss = use_reproj3d_loss
        self.reproj3d_weight = reproj3d_weight
        self.reproj3d_loss_type = reproj3d_loss_type
        self.reproj3d_normalize = reproj3d_normalize
        self.reproj3d_update_calib = reproj3d_update_calib
        self.reproj3d_warmup_steps = reproj3d_warmup_steps
        self.bev3d_grad_scale = bev3d_grad_scale
        self.attr3d_kwargs = dict(
            attr3d_neck_dim=attr3d_neck_dim,
            attr3d_head_dim=attr3d_head_dim,
            attr3d_points=attr3d_points,
            attr3d_max_offset_cells=attr3d_max_offset_cells,
            attr3d_detach_trunk=attr3d_detach_trunk,
        )
        self.size_scale_cm = size_scale_cm
        self.calib_warmup_steps = calib_warmup_steps
        self.calib_warmup_mode = calib_warmup_mode
        self.calib_warmup_loss_threshold = calib_warmup_loss_threshold
        self._calib_unfrozen = False
        self._latest_center_loss = float('inf')
        # Guards against re-resolving calib_warmup_steps /
        # reproj3d_warmup_steps more than once (e.g. multiple .fit()
        # calls, or resuming from a checkpoint).
        self._warmup_steps_resolved = False

        # Does this run's dataset carry real 3D GT?
        self.has_3d_gt = False
        self._has_3d_resolved = False
        self.unlabeled_use_3d = bool(unlabeled_use_3d)
        # Per-sequence GT routing, filled by _resolve_has_3d_gt().
        # Sequences absent from these dicts default to annotated-2D,
        # which reproduces the pre-change behaviour exactly.
        self._seq_has_gt = {}          # seq_num -> has 2D annotations
        self._seq_has_3d = {}          # seq_num -> has 3D boxes
        self._test_unlabeled_seqs = []
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
        # Inference-only sequences (no GT): predictions are routed here
        # and exported to *_unlabeled.txt. No metric ever reads these.
        self.moda_pred_unlabeled_list = []
        self.mota_pred_unlabeled_list = []
        self.moda_pred_3d_unlabeled_list = []
        self.mota_pred_3d_unlabeled_list = []
        self.obb_pred_mota_unlabeled = []

        # Validation MODA bookkeeping (checkpoint selection)
        # Separate buffers from the test-time ones so the test loop is
        # completely untouched. Pooled across ALL val sequences.
        self.val_moda_interval = val_moda_interval
        self.val_moda_gt_list, self.val_moda_pred_list = [], []
        self._compute_val_moda = False  # set per-epoch in on_validation_epoch_start
        self._val_epoch_idx = 0  # counts real (non-sanity) val epochs
        self._last_val_moda = 0.0  # fallback value on skipped epochs

        self.nms_score_threshold = float(nms_score_threshold)
        self.aux3d_weight = float(aux3d_weight)
        self.aux3d_warmup_steps = float(aux3d_warmup_steps)
        self.aux3d_logvar_min = float(aux3d_logvar_min)
        self.aux3d_logvar_max = float(aux3d_logvar_max)
        self.yaw_mag_weight = float(yaw_mag_weight)
        self.size_prior_weight = float(size_prior_weight)
        self.val_conf_sweep = bool(val_conf_sweep)
        self.val_moda_gate_cells = float(val_moda_gate_cells)
        self.track_assoc_dist_cm = float(track_assoc_dist_cm)
        self.track_dup_dist_cm = float(track_dup_dist_cm)
        self.track_buffer = int(track_buffer)
        self._last_val_conf = float(conf_threshold)

        self.track_lying_velocity_freeze = bool(track_lying_velocity_freeze)
        self._tracker_times = []  # tracker-only ms per frame

        # 3D-aware tracking
        self.track_use_3d_attrs = bool(track_use_3d_attrs)
        self.track_attr_cfg = dict(
            size_mode=str(track_size_mode),
            size_alpha=float(track_size_alpha),
            size_reject_cm=float(track_size_reject_cm),
            median_window=int(track_size_median_window),
            yaw_mode=str(track_yaw_mode),
            yaw_alpha=float(track_yaw_alpha),
            flip_align=bool(track_yaw_flip_align),
            yaw_q=float(track_yaw_q),
            yaw_r=float(track_yaw_r),
            posture_alpha=float(track_posture_alpha),
            posture_hysteresis=float(track_posture_hysteresis),
        )
        # diagnostics (test time)
        self.track3d_smoothed = 0
        self.track3d_fallback = 0
        self.track3d_yaw_flips = 0

        self.track_writeback_to_detections = track_writeback_to_detections
        # track-consistency loss
        self.use_track_attr_consistency = bool(use_track_attr_consistency)
        self.track_attr_consistency_weight = float(track_attr_consistency_weight)
        self.track_yaw_consistency_weight = float(track_yaw_consistency_weight)
        self.track_assoc_use_iou = bool(track_assoc_use_iou)
        self.track_assoc_iou_weight = float(track_assoc_iou_weight)
        self._assoc_iou_active = False
        self.track_attr_bank_alpha = float(track_attr_bank_alpha)
        self.track_attr_bank_max_gap = int(track_attr_bank_max_gap)
        self.track_attr_consistency_warmup_steps = float(
            track_attr_consistency_warmup_steps)
        # {(seq, cow_id): {'size': (3,) tensor, 'yaw': (2,) tensor,
        #                  'frame': int, 'step': int}}  -- all DETACHED
        self._attr_bank = {}

        # enriched 3D outputs + OBB GT/pred
        self.eval_match_dist_cm = eval_match_dist_cm
        self.moda_pred_3d_list, self.mota_pred_3d_list = [], []
        self.obb_gt_moda, self.obb_pred_moda = [], []
        self.obb_gt_mota, self.obb_pred_mota = [], []
        self.mota_gt_3d_list = []  # GT tracks + yaw/L/W/H/posture (cm)
        self.posture_correct, self.posture_total = 0, 0
        self.yaw_errors, self.dim_errors = [], []
        # OBB evaluation config + track-box coverage diagnostic
        self.obb_iou_thresholds = tuple(
            float(t) for t in (obb_iou_thresholds or (0.25, 0.50))
        )
        self.obb_tracks_total = 0
        self.obb_tracks_missing_box = 0
        self.obb_nms_iou = float(obb_nms_iou)
        self._grid_cell_cm = None  # cm per world-grid cell, captured once

        # canonical OBB export via utils.obb.decode_obb
        # Strictly additive; never read by any metric computation.
        self.obb_export_list = []
        self._obb_export_has3d = None  # schema latched on the first batch
        self._obb_export_warned = False

        self.frame = 0
        self.test_tracker = None
        self._current_test_seq = -1
        self.inference_times = []
        self.test_start_time = None

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
                bev3d_grad_scale=self.bev3d_grad_scale,
                swin_pretrained_path=self.swin_pretrained_path,
                swin_freeze_backbone=self.swin_freeze_backbone,
                img_grad_checkpoint=self.img_grad_checkpoint,
                **self.attr3d_kwargs,
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
            # freeze if warmup is requested
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
            'reset_calibration_at_test',
            'calibration_dir',
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

    def _aux3d_ramp(self):
        """Linear 0 -> 1 ramp of the 3D auxiliary losses over the first
        aux3d_warmup_steps optimiser steps (a value in (0,1) is resolved
        to a fraction of total steps in on_fit_start)."""
        if self.aux3d_warmup_steps <= 0:
            return 1.0
        return float(min(1.0, self.global_step
                         / float(self.aux3d_warmup_steps)))

    # ==================================================================
    # Per-object 3D attribute queries (query-style head)
    # ==================================================================
    def _attr_fn_3d(self, output):
        """Build the decode-time attribute callback for THIS output dict.

        Returns None (2D-only run / old checkpoint) or a callable
        (B,K,2) mem coords -> decode-compatible attribute dict.
        The callback closes over output['attr_feat'], so for a
        batch-sliced decode pass a sliced dict:
        ``self._attr_fn_3d({'attr_feat': output['attr_feat'][b:b+1]})``.
        """
        head = getattr(getattr(self.model, 'decoder', None),
                       'attr3d_head', None)
        feat = output.get('attr_feat')
        if head is None or feat is None:
            return None
        size_scale = float(self.size_scale_cm)

        def _fn(xy):
            return head.query_extra(feat, xy, size_scale=size_scale)
        return _fn

    def _query_gt_attrs(self, output, item):
        """Run the query head at every GT object location (teacher forcing).

        MEMORY: only VALID rows (cow_id > 0 AND in-bounds) are queried.
        The old version ran the head at all M = max_objects (60) padded
        rows per batch element, i.e. ~6x more grid_sample points and MLP
        activations than there are supervised cows. Querying is
        point-wise (bilinear gather + per-point MLP; the neck's
        InstanceNorm runs on the dense map and is untouched), so
        compacting first is bit-identical for every supervised row.

        Returns None when there is nothing to supervise, else a dict:
            yaw_pred  : (N, 2) raw (sin 2t, cos 2t), N = total valid rows
            size_pred : (N, 3) metres
            post_pred : (N,)   standing logits
            yaw_gt    : (N, 2) GT (sin 2t, cos 2t)
            size_gt   : (N, 3) GT (l, w, h) in METRES
            post_gt   : (N,)   GT standing probability
            valid     : (B, M) bool  -- kept for the consistency loss
            per_batch : list[B] of {'yaw','size','posture'} compacted to
                        the valid rows -- aligned with the row selection
                        reprojection_loss_3d recomputes from grid_gt_3d.
        """
        g3d = item['grid_gt_3d']                          # (B, M, 11)
        head = getattr(getattr(self.model, 'decoder', None),
                       'attr3d_head', None)
        feat = output.get('attr_feat')
        if head is None or feat is None or g3d.shape[-1] < 11:
            return None
        B, M = g3d.shape[:2]
        Y, X = feat.shape[-2], feat.shape[-1]
        xy = g3d[..., 0:2].float()
        inb = ((xy[..., 0] >= 0) & (xy[..., 0] < float(X))
               & (xy[..., 1] >= 0) & (xy[..., 1] < float(Y)))
        valid = (g3d[..., 5] > 0) & inb
        if not bool(valid.any()):
            return None

        yaw_gt_full = torch.stack(
            (torch.sin(2.0 * g3d[..., 6]), torch.cos(2.0 * g3d[..., 6])),
            dim=-1)
        size_gt_full = g3d[..., 7:10] / float(self.size_scale_cm)  # cm -> m
        post_gt_full = g3d[..., 10]

        # Query ONLY the valid rows (per batch element), then concatenate.
        yaw_p, size_p, post_p = [], [], []
        yaw_g, size_g, post_g = [], [], []
        per_batch = []
        for b in range(B):
            vb = valid[b]
            if not bool(vb.any()):
                per_batch.append({'yaw': feat.new_zeros((0, 2)),
                                  'size': feat.new_zeros((0, 3)),
                                  'posture': feat.new_zeros((0, 1))})
                continue
            raw_b = head.query(feat[b:b + 1], xy[b][vb].unsqueeze(0))
            per_batch.append({'yaw': raw_b['yaw'][0],
                              'size': raw_b['size'][0],
                              'posture': raw_b['posture'][0]})
            yaw_p.append(raw_b['yaw'][0])
            size_p.append(raw_b['size'][0])
            post_p.append(raw_b['posture'][0].squeeze(-1))
            yaw_g.append(yaw_gt_full[b][vb])
            size_g.append(size_gt_full[b][vb])
            post_g.append(post_gt_full[b][vb])

        return {'yaw_pred': torch.cat(yaw_p),
                'size_pred': torch.cat(size_p),
                'post_pred': torch.cat(post_p),
                'yaw_gt': torch.cat(yaw_g),
                'size_gt': torch.cat(size_g),
                'post_gt': torch.cat(post_g),
                'valid': valid, 'per_batch': per_batch}

    # ==================================================================
    # Track-consistency loss: a prediction must agree with the SAME
    # animal's own recent history.
    # ==================================================================
    def _prune_attr_bank(self, max_entries=8192, max_age_steps=500):
        if len(self._attr_bank) <= max_entries:
            return
        cutoff = int(self.global_step) - int(max_age_steps)
        self._attr_bank = {k: v for k, v in self._attr_bank.items()
                           if v['step'] >= cutoff}

    def _track_attr_consistency_loss(self, item, attr3d):
        """Return (size_terms, yaw_terms) as lists of scalar losses.

        Same EMA self-consistency as before, but the predictions are the
        per-object query head's outputs at the GT centres (taken from
        `_query_gt_attrs`, so this costs zero extra head evaluations and
        is consistent with what the aux3d losses supervise).

        `track_attr_bank_max_gap` drops stale entries (sequence switch,
        epoch restart, occlusion gap).
        """
        g3d = item['grid_gt_3d']
        frames = item['frame'].detach().cpu().reshape(-1).long().tolist()
        seqs = item.get('sequence_num',
                        torch.zeros_like(item['frame'])
                        ).detach().cpu().reshape(-1).long().tolist()

        a = self.track_attr_bank_alpha
        gap = self.track_attr_bank_max_gap
        want_yaw = self.track_yaw_consistency_weight > 0.0
        size_terms, yaw_terms = [], []

        for b in range(g3d.shape[0]):
            mask = attr3d['valid'][b]
            if not bool(mask.any()):
                continue
            rows_v = g3d[b][mask]  # supervised GT rows
            preds = attr3d['per_batch'][b]  # aligned with rows_v
            f = int(frames[b])
            seq = int(seqs[b])
            for i in range(rows_v.shape[0]):
                key = (seq, int(rows_v[i, 5].item()))
                s_pred = preds['size'][i]  # (3,)
                v_pred = None
                if want_yaw:
                    v = preds['yaw'][i]
                    v_pred = v / v.norm().clamp(min=1e-3)  # (2,)

                entry = self._attr_bank.get(key)
                fresh = (entry is not None
                         and 0 < (f - entry['frame']) <= gap)

                if fresh:
                    # Bank entries live on the CPU (see below); move the
                    # (tiny) target back to the prediction's device.
                    size_terms.append(torch.nn.functional.smooth_l1_loss(
                        s_pred, entry['size'].to(s_pred.device),
                        reduction='sum'))
                    if want_yaw and entry['yaw'] is not None:
                        yaw_terms.append(torch.nn.functional.smooth_l1_loss(
                            v_pred, entry['yaw'].to(v_pred.device),
                            reduction='sum'))

                # update the (detached) bank
                new_s = s_pred.detach().cpu()
                new_v = None if v_pred is None else v_pred.detach().cpu()
                if not fresh:
                    self._attr_bank[key] = {
                        'size': new_s, 'yaw': new_v,
                        'frame': f, 'step': int(self.global_step)}
                else:
                    es = entry['size'] * (1.0 - a) + new_s * a
                    ev = entry['yaw']
                    if ev is not None and new_v is not None:
                        ev = ev * (1.0 - a) + new_v * a  # no flip-align
                        ev = ev / ev.norm().clamp(min=1e-3)
                    else:
                        ev = new_v
                    self._attr_bank[key] = {
                        'size': es, 'yaw': ev,
                        'frame': f, 'step': int(self.global_step)}

        self._prune_attr_bank()
        return size_terms, yaw_terms

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

        attr3d = None  # filled by the per-object query block below

        # ──────────────────────────────────────────────────────────
        # 3D attribute losses — per-OBJECT queries (replaces the dense
        # per-pixel yaw/size/posture supervision on yaw_bev/size_bev/
        # posture_bev; those painted targets are no longer consumed).
        # ──────────────────────────────────────────────────────────
        if self.has_3d_gt and item is not None and 'grid_gt_3d' in item:
            attr3d = self._query_gt_attrs(output, item)
        if attr3d is not None:
            # attr3d tensors are COMPACT: (N, ...) over valid rows only.
            # A 0/1-masked reduce_masked_mean over valid rows is exactly
            # a plain mean over those rows -- identical loss values.

            # ── (1) yaw: direction + magnitude gauge, per object ──
            yaw_norm = attr3d['yaw_pred'].norm(dim=-1, keepdim=True)  # (N,1)
            yaw_n = attr3d['yaw_pred'] / yaw_norm.clamp(min=1e-3)
            yaw_loss = torch.nn.functional.smooth_l1_loss(
                yaw_n, attr3d['yaw_gt'], reduction='none'
            ).sum(dim=-1).mean()
            yaw_mag_loss = ((yaw_norm - 1.0) ** 2).mean()

            # ── (2) size, per object (metres) ──
            size_loss = torch.nn.functional.smooth_l1_loss(
                attr3d['size_pred'], attr3d['size_gt'], reduction='none'
            ).sum(dim=-1).mean()

            # ── (3) posture, per object ──
            posture_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                attr3d['post_pred'], attr3d['post_gt'], reduction='none'
            ).mean()

            loss_dict['yaw_loss'] = yaw_loss
            loss_dict['size_loss'] = size_loss
            loss_dict['posture_loss'] = posture_loss
            loss_dict['yaw_mag_loss'] = yaw_mag_loss

            lo, hi = self.aux3d_logvar_min, self.aux3d_logvar_max
            yaw_w = self.model.yaw_weight.clamp(lo, hi)
            size_w = self.model.size_weight.clamp(lo, hi)
            post_w = self.model.posture_weight.clamp(lo, hi)
            w3 = self.aux3d_weight * self._aux3d_ramp()

            loss_weight_dict['yaw_loss'] = w3 * (
                    torch.exp(-yaw_w) * yaw_loss + yaw_w)
            loss_weight_dict['size_loss'] = w3 * (
                    torch.exp(-size_w) * size_loss + size_w)
            loss_weight_dict['posture_loss'] = w3 * (
                    torch.exp(-post_w) * posture_loss + post_w)
            loss_weight_dict['yaw_mag_loss'] = (
                    self.yaw_mag_weight * self._aux3d_ramp() * yaw_mag_loss)

            # log-only (NOT summed into total_loss)
            loss_dict['yaw_logvar'] = self.model.yaw_weight.detach()
            loss_dict['size_logvar'] = self.model.size_weight.detach()
            loss_dict['posture_logvar'] = self.model.posture_weight.detach()

        # ──────────────────────────────────────────────────────────
        # track-consistency (temporal self-consistency) loss
        # ──────────────────────────────────────────────────────────
        if (self.use_track_attr_consistency and self.training
                and attr3d is not None
                and self.global_step >= self.track_attr_consistency_warmup_steps):
            s_terms, y_terms = self._track_attr_consistency_loss(item, attr3d)
            if s_terms:
                cons = torch.stack(s_terms).mean()
                loss_dict['track_size_cons'] = cons
                loss_weight_dict['track_size_cons'] = (
                        self.track_attr_consistency_weight * cons)
            if y_terms:
                yc = torch.stack(y_terms).mean()
                loss_dict['track_yaw_cons'] = yc
                loss_weight_dict['track_yaw_cons'] = (
                        self.track_yaw_consistency_weight * yc)

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

        # 3D-box corner reprojection loss
        if (self.use_reproj3d_loss and item is not None
                and 'grid_gt_3d' in item
                and attr3d is not None
                and self.global_step >= self.reproj3d_warmup_steps):

            if self.learn_calibration:
                ext_for_3d = self.cal_refine(item['extrinsic'])
                if not (self.reproj3d_update_calib and self.use_reproj_loss):
                    ext_for_3d = ext_for_3d.detach()
                    if self.reproj3d_update_calib and self.global_step == 0:
                        print("[CALIB] reproj3d_update_calib=True but "
                              "use_reproj_loss=False -> extrinsics DETACHED "
                              "inside the 3D loss (arm-symmetry guard).")
            else:
                ext_for_3d = item['extrinsic']

            reproj3d, n3d = reprojection_loss_3d(
                item['grid_gt_3d'],
                item['img_gt_2d'],
                item['intrinsic_original'],
                ext_for_3d,
                [pb['yaw'] for pb in attr3d['per_batch']],
                [pb['size'] for pb in attr3d['per_batch']],
                (self.Y, self.X),
                loss_type=self.reproj3d_loss_type,
                size_scale=self.size_scale_cm,
                normalize=self.reproj3d_normalize,
            )
            loss_dict['reproj3d_loss'] = self.reproj3d_weight * reproj3d
            loss_dict['reproj3d_nterms'] = torch.tensor(
                float(n3d), device=reproj3d.device)
            loss_weight_dict['reproj3d_loss'] = self.reproj3d_weight * reproj3d

        total_loss = (sum(loss_weight_dict.values())
                      + sum(stats_dict.values()))

        # L2 regularisation on calibration parameters
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
        # track center loss for warmup
        if self.learn_calibration and not self._calib_unfrozen:
            self._latest_center_loss = (
                loss_dict['center_loss'].detach().item()
            )

        return total_loss

    def validation_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        if batch_idx % 100 == 1:
            self.plot_data(target, output, item, batch_idx)

        total_loss, loss_dict = self.loss(target, output, item)

        B = item['img'].shape[0]
        self.log('val_loss', total_loss, batch_size=B, sync_dist=True)
        self.log(
            'val_center', loss_dict['center_loss'],
            batch_size=B, sync_dist=True,
        )
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, batch_size=B, sync_dist=True)

        # buffer decoded detections for pooled val MODA.
        if self._compute_val_moda:
            self._accumulate_val_moda(item, output)

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
        # val-MODA schedule
        sanity = bool(
            getattr(self.trainer, 'sanity_checking', False)
        ) if self.trainer is not None else False

        self._compute_val_moda = (
            (not sanity)
            and self.val_moda_interval > 0
            and (self._val_epoch_idx % self.val_moda_interval == 0)
        )
        self.val_moda_gt_list, self.val_moda_pred_list = [], []

        # per-camera reprojection accumulators
        if not self.learn_calibration:
            return
        S = self.cal_refine.num_cameras
        self._val_reproj_n = defaultdict(int)
        self._val_reproj_sum = defaultdict(float)
        self._val_reproj_sum_r = defaultdict(float)

    def on_validation_epoch_end(self):
        # pooled validation MODA
        if self._compute_val_moda:
            self._last_val_moda, self._last_val_conf = \
                self._compute_pooled_val_moda()
            self.val_moda_gt_list, self.val_moda_pred_list = [], []
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        sanity = bool(
            getattr(self.trainer, 'sanity_checking', False)
        ) if self.trainer is not None else False

        if not sanity:
            self.log('val_moda', float(self._last_val_moda),
                     prog_bar=True, sync_dist=False)
            self.log('val_best_conf', float(self._last_val_conf),
                     sync_dist=False)
            self._val_epoch_idx += 1

        # per-camera reprojection error
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
    def _accumulate_val_moda(self, item, output):
        """Buffer EVERY surviving peak + its score so the confidence
        threshold can be swept at epoch end.
        """
        dev = self.device.type
        with torch.autocast(device_type=dev, enabled=False):
            center = output['instance_center'].float().sigmoid()
            offset = output['instance_offset'].float()
            kw = dict(K=self.max_detections,
                      size_scale=self.size_scale_cm,
                      **self._nms_kwargs(item))
            kw['nms_iou'] = 0.0
            decoded = decode.decoder(
                center, offset, None,
                attr_fn=None,
                **kw)
            if self._grid_cell_cm is None and 'worldcoord_from_worldgrid' in item:
                W = item['worldcoord_from_worldgrid'].detach().float().cpu()
                if W.dim() == 3:
                    W = W[0]
                self._grid_cell_cm = abs(float(W[0, 0]))
            xy_e, _, scores_e = decoded[0], decoded[1], decoded[2]
            ref_xy = obb.mem_to_ref_xy(
                self.vox_util, xy_e.float(), self.Y, self.Z, self.X).cpu()

        seq_nums = item.get('sequence_num', torch.zeros_like(item['frame']))
        sc = scores_e.float().cpu()
        for b, (frame, seq_num, grid_gt) in enumerate(
                zip(item['frame'], seq_nums, item['grid_gt'])):
            gf = int(seq_num.item()) * 1_000_000 + int(frame.item())
            self.val_moda_gt_list.extend(
                [[gf, float(x), float(y)]
                 for x, y, _ in grid_gt[grid_gt.sum(1) != 0]])
            self.val_moda_pred_list.extend(
                [[gf, float(ref_xy[b, k, 0]), float(ref_xy[b, k, 1]),
                  float(sc[b, k])]
                 for k in range(sc.shape[1]) if sc[b, k] > 0.0])

    def _compute_pooled_val_moda(self):
        if not self.val_moda_pred_list or not self.val_moda_gt_list:
            print("[val_moda] no detections -> val_moda = 0.0")
            return 0.0, float(self.conf_threshold)

        gt = np.asarray(self.val_moda_gt_list, dtype=np.float64)
        pr = np.asarray(self.val_moda_pred_list, dtype=np.float64)
        gt_by_f = {int(f): gt[gt[:, 0] == f][:, 1:3] for f in np.unique(gt[:, 0])}
        pr_by_f = {int(f): pr[pr[:, 0] == f][:, 1:4] for f in np.unique(pr[:, 0])}

        # Same 100 cm gate as the test-time CLEAR-MOD (and as MOTA's
        # max_d2=1 in metres), so the checkpoint we select is the best
        # one under the metric we actually report.
        gate = (100.0 / self._grid_cell_cm
                if self._grid_cell_cm else float(self.val_moda_gate_cells))
        thrs = (np.round(np.arange(0.05, 0.96, 0.05), 3)
                if self.val_conf_sweep
                else np.array([float(self.conf_threshold)]))
        rows = [self._mod_at_threshold(gt_by_f, pr_by_f, float(t), gate)
                for t in thrs]

        best = max(rows, key=lambda r: r['moda'])
        ref = min(rows, key=lambda r: abs(r['thr'] - self.conf_threshold))

        # threshold-free summary (interpolated AP over the sweep)
        r_ = np.array([x['recall'] for x in rows]) / 100.0
        p_ = np.array([x['precision'] for x in rows]) / 100.0
        o = np.argsort(r_); r_, p_ = r_[o], p_[o]
        p_ = np.maximum.accumulate(p_[::-1])[::-1]
        ap = 100.0 * float(np.sum(np.diff(np.concatenate([[0.0], r_])) * p_))

        self.log('val_ap', ap, sync_dist=False)
        self.log('val_moda_at_conf', ref['moda'], sync_dist=False)
        self.log('val_recall', best['recall'], sync_dist=False)
        self.log('val_precision', best['precision'], sync_dist=False)
        self.log('val_modp', best['modp'], sync_dist=False)

        print(f"\n[val_moda] epoch {self.current_epoch}: "
              f"MODA*={best['moda']:.2f}% @thr={best['thr']:.2f} "
              f"(R{best['recall']:.1f} P{best['precision']:.1f} "
              f"MODP{best['modp']:.1f}) | AP={ap:.2f} | "
              f"MODA@{self.conf_threshold:.2f}={ref['moda']:.2f} | "
              f"GT={len(gt)} peaks={len(pr)}\n")
        return float(best['moda']), float(best['thr'])

    @torch.no_grad()
    def _accumulate_reproj_diagnostics(self, item):
        """
        For each camera, compute pixel reprojection error using
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
            pids = gc[:, 2].long()
            N = pids.shape[0]

            Wb = W[b]
            # world-grid -> world cm (shared helper, identical arithmetic)
            w_xy = obb.worldgrid_to_worldcm(gc[:, :2].float(), Wb)
            wx, wy = w_xy[:, 0], w_xy[:, 1]

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

    # ==================================================================
    # 3D-GT availability (resolved once per run)
    # ==================================================================
    @staticmethod
    def _dataset_splits_from_datamodule(dm):
        """Return every split dataset that currently exists on *dm*.

        Mirrors PedestrianDataModule.get_base_extrinsics(): depending on
        the stage only some of train/val/test/predict are populated.
        """
        splits = []
        for attr in ('data_train', 'data_val',
                     'data_test', 'data_predict'):
            ds = getattr(dm, attr, None)
            if ds is not None:
                splits.append(ds)
        return splits

    def _resolve_has_3d_gt(self):
        """Resolve ``self.has_3d_gt`` AND the per-sequence GT flags.
        """
        if self._has_3d_resolved:
            return self.has_3d_gt

        try:
            trainer = self.trainer
        except RuntimeError:  # module not attached to a Trainer
            trainer = None
        dm = getattr(trainer, 'datamodule', None) if trainer else None
        if dm is None:
            # Nothing to inspect yet -- stay unresolved so a later hook
            # (on_fit_start / on_test_start) can try again.
            return self.has_3d_gt

        flags = []  # (sequence_num | None, has_gt, has_3d)
        for ds in self._dataset_splits_from_datamodule(dm):
            subsets = (ds.datasets
                       if (hasattr(ds, 'datasets') and ds.datasets)
                       else [ds])
            for sub in subsets:
                base = getattr(sub, 'base', None)
                if base is None:
                    continue
                flags.append((
                    getattr(sub, 'sequence_num', None),
                    bool(getattr(base, 'has_gt', True)),
                    bool(getattr(base, 'has_3d', False)),
                ))

        if not flags:
            # setup() has not built any split yet -- try again later.
            return self.has_3d_gt

        self._seq_has_gt = {int(s): g for s, g, _ in flags
                            if s is not None}
        self._seq_has_3d = {int(s): h for s, _, h in flags
                            if s is not None}
        self._test_unlabeled_seqs = sorted(
            s for s, g in self._seq_has_gt.items() if not g)

        n_all = len(flags)
        n_gt = sum(g for _, g, _ in flags)
        n_3d_annotated = sum(h for _, g, h in flags if g)

        if n_gt == 0:
            # Inference-only test set: nothing to evaluate, no dataset
            # signal to resolve has_3d_gt from -> config flag decides.
            self.has_3d_gt = bool(self.unlabeled_use_3d)
            print(f"[GT-ROUTING] ALL {n_all} sequence(s) are UNLABELLED "
                  f"(no annotations_positions/). Detection/tracking "
                  f"metrics will be SKIPPED; predictions go to the "
                  f"*_unlabeled.txt exports.")
            print(f"[GT-ROUTING] has_3d_gt = {self.has_3d_gt} (from "
                  f"unlabeled_use_3d; no GT to resolve it from)")
        else:
            self.has_3d_gt = bool(n_3d_annotated > 0)
            print(f"[GT-ROUTING] has_3d_gt = {self.has_3d_gt} "
                  f"({n_3d_annotated}/{n_gt} annotated sequence(s) "
                  f"expose 3D boxes)")

        if self._test_unlabeled_seqs:
            print(f"[GT-ROUTING] inference-only sequences: "
                  f"{self._test_unlabeled_seqs} -- excluded from EVERY "
                  f"metric (moda/mota/obb); predictions exported to "
                  f"*_unlabeled.txt and the canonical obb_detections_*.")
        if 0 < n_3d_annotated < n_gt:
            print(f"[GT-ROUTING] NOTE: mixed 2D/3D annotations among "
                  f"annotated sequences. 3D metric streams are gated PER "
                  f"SEQUENCE: 2D-annotated sequences contribute "
                  f"point-MODA/MOTA only.")

        self._has_3d_resolved = True

        if n_3d_annotated == 0 and self.use_reproj3d_loss:
            print("[OBB] no annotated sequence with 3D boxes -> "
                  "use_reproj3d_loss disabled for this run.")
            self.use_reproj3d_loss = False

        return self.has_3d_gt

    def setup(self, stage=None):
        """LightningModule.setup runs *after* datamodule.setup(stage),
        so the split datasets already exist here for fit / validate /
        test / predict."""
        super().setup(stage)
        self._resolve_has_3d_gt()

    def on_fit_start(self):
        self._resolve_has_3d_gt()

        if self._warmup_steps_resolved:
            return
        if self.trainer is None:
            return
        total_steps = int(self.trainer.estimated_stepping_batches)
        if total_steps <= 0:
            return
        if 0 < self.calib_warmup_steps < 1.0:
            frac = self.calib_warmup_steps
            self.calib_warmup_steps = max(1, int(frac * total_steps))
            print(f"[WARMUP] calib_warmup_steps: fraction {frac} "
                  f"-> {self.calib_warmup_steps} steps "
                  f"(total_steps={total_steps})")
        if 0 < self.reproj3d_warmup_steps < 1.0:
            frac = self.reproj3d_warmup_steps
            self.reproj3d_warmup_steps = max(1, int(frac * total_steps))
            print(f"[WARMUP] reproj3d_warmup_steps: fraction {frac} "
                  f"-> {self.reproj3d_warmup_steps} steps "
                  f"(total_steps={total_steps})")
        if 0 < self.aux3d_warmup_steps < 1.0:
            frac = self.aux3d_warmup_steps
            self.aux3d_warmup_steps = max(1, int(frac * total_steps))
            print(f"[WARMUP] aux3d_warmup_steps: fraction {frac} "
                  f"-> {self.aux3d_warmup_steps} steps "
                  f"(total_steps={total_steps})")

        if 0 < self.track_attr_consistency_warmup_steps < 1.0:
            frac = self.track_attr_consistency_warmup_steps
            self.track_attr_consistency_warmup_steps = max(
                1, int(frac * total_steps))
            print(f"[WARMUP] track_attr_consistency_warmup_steps: "
                  f"fraction {frac} -> "
                  f"{self.track_attr_consistency_warmup_steps} steps "
                  f"(total_steps={total_steps})")

        self._warmup_steps_resolved = True

    def on_train_epoch_start(self):
        """Drop the per-cow attribute history at every epoch boundary:
        frame indices restart, so keeping it would pair frame N of the
        old epoch with frame 0 of the new one."""
        self._attr_bank = {}

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
            'reset_calibration_at_test',
            'calibration_dir',
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
        self._resolve_has_3d_gt()

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

        # Mode 1: Reset to zero (ablation)
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

        # Mode 2: Load from directory
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

        # Mode 3: Use checkpoint (default)
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
        # Created on the first test frame, after decode/_nms_kwargs has
        # resolved the real BEV cell size. This removes the 10 cm fallback.
        self.test_tracker = None
        self.moda_gt_list, self.moda_pred_list = [], []
        self.mota_gt_list, self.mota_pred_list = [], []
        # inference-only sequences (no GT): separate export streams
        self.moda_pred_unlabeled_list = []
        self.mota_pred_unlabeled_list = []
        self.moda_pred_3d_unlabeled_list = []
        self.mota_pred_3d_unlabeled_list = []
        self.obb_pred_mota_unlabeled = []
        self._test_unlabeled_seqs = sorted(
            s for s, g in self._seq_has_gt.items() if not g)
        self.mota_gt_3d_list = []
        self.moda_pred_3d_list, self.mota_pred_3d_list = [], []
        self.obb_gt_moda, self.obb_pred_moda = [], []
        self.obb_gt_mota, self.obb_pred_mota = [], []
        self.posture_correct, self.posture_total = 0, 0
        self.yaw_errors, self.dim_errors = [], []
        self.obb_tracks_total = 0
        self.obb_tracks_missing_box = 0
        self._grid_cell_cm = None
        self.obb_export_list = []
        self._obb_export_has3d = None
        self._obb_export_warned = False
        self.track3d_smoothed = 0
        self.track3d_fallback = 0
        self.track3d_yaw_flips = 0
        self._tracker_times = []
        self.inference_times = []
        self.test_start_time = None

    # ==================================================================
    # 3D evaluation helpers
    # ==================================================================
    @staticmethod
    def _wrap_pi(a):
        """Wrap an angle (or array of angles) into [-pi, pi]."""
        return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _wrap_half_pi(a):
        """Wrap an angle difference to [-pi/2, pi/2) -- the true period of
        an oriented box (theta and theta + pi are the same box)."""
        return (np.asarray(a) + np.pi / 2) % np.pi - np.pi / 2

    @staticmethod
    def _greedy_match(pred_xy, gt_xy, thresh):
        """Greedy nearest-neighbour matching in the given prediction order
        (call with predictions pre-sorted by descending score).
        Returns a list of (pred_idx, gt_idx) pairs, each GT used at most once.
        """
        matches = []
        pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
        gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
        if len(pred_xy) == 0 or len(gt_xy) == 0:
            return matches
        taken = np.zeros(len(gt_xy), dtype=bool)
        for pi in range(len(pred_xy)):
            d = np.linalg.norm(gt_xy - pred_xy[pi], axis=1)
            d[taken] = np.inf
            gi = int(np.argmin(d))
            if d[gi] <= thresh:
                taken[gi] = True
                matches.append((pi, gi))
        return matches

    @staticmethod
    def _mod_at_threshold(gt_by_f, pred_by_f, thr, gate):
        """Hungarian CLEAR-MOD at one confidence threshold.

        Identical arithmetic to tools/eval_sweep.py so that the number
        logged during training and the number produced offline are
        directly comparable. `gate` is in world-GRID cells.
        """
        from scipy.optimize import linear_sum_assignment
        TP = FP = FN = 0
        dsum, dn = 0.0, 0
        for f, G in gt_by_f.items():
            P = pred_by_f.get(f)
            P = (P[P[:, 2] > thr][:, :2]
                 if (P is not None and len(P)) else np.zeros((0, 2)))
            m = 0
            if len(P) and len(G):
                C = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
                Cx = np.where(C <= gate, C, 1e6)
                ri, ci = linear_sum_assignment(Cx)
                keep = Cx[ri, ci] < 1e6
                m = int(keep.sum())
                dsum += float(C[ri[keep], ci[keep]].sum());
                dn += m
            TP += m;
            FP += len(P) - m;
            FN += len(G) - m
        nG = max(TP + FN, 1)
        return dict(
            thr=float(thr), tp=TP, fp=FP, fn=FN,
            recall=100.0 * TP / nG,
            precision=100.0 * TP / max(TP + FP, 1),
            moda=100.0 * (1.0 - (FP + FN) / nG),
            modp=(100.0 * (1.0 - (dsum / dn) / gate)) if dn else 0.0,
        )

    @torch.no_grad()
    def _sample_query_3d(self, attr_feat_b, mx, my):
        """Per-object replacement for the old dense-cell read.

        attr_feat_b : (1, C, Y, X) attribute-neck features of ONE batch
                      element; mx, my : FLOAT BEV memory coords
                      (sub-pixel -- no rounding).
        Returns (yaw_rad, dims_cm (3,), posture_class int).
        """
        head = self.model.decoder.attr3d_head
        xy = torch.tensor([[[float(mx), float(my)]]],
                          dtype=torch.float32, device=attr_feat_b.device)
        e = head.query_extra(attr_feat_b, xy,
                             size_scale=float(self.size_scale_cm))
        yaw = float(e['yaw_angle'][0, 0])
        dims = e['dimensions'][0, 0].cpu().float().numpy().astype(np.float64)
        post = int(e['posture_class'][0, 0])
        return yaw, dims, post

    def _bev_cell_size_cm(self, item):
        """Centimetres per BEV *memory* cell.

        cm/mem-cell = (cm per world-grid cell)  x  (grid cells per mem cell)
                    =  W[0,0]                   x  vox_util.default_vox_size_X

        For mmCows (grid_cell_size=10, bounds X 0..192 over X=192 mem
        cells) this is 10.0 cm. Cached after the first batch.
        """
        if self._bev_cell_cm is not None:
            return self._bev_cell_cm
        cm_per_grid = 1.0
        if item is not None and 'worldcoord_from_worldgrid' in item:
            W = item['worldcoord_from_worldgrid'].detach().float().cpu()
            if W.dim() == 3:
                W = W[0]
            cm_per_grid = abs(float(W[0, 0])) or 1.0
        grid_per_mem = abs(float(self.vox_util.default_vox_size_X)) or 1.0
        self._bev_cell_cm = cm_per_grid * grid_per_mem
        if self.nms_radius_cm > 0 or self.nms_iou > 0 or self.nms_kernel > 3:
            print(f"[NMS] 1 BEV cell = {self._bev_cell_cm:.2f} cm | "
                  f"kernel={self.nms_kernel} "
                  f"(+/-{(self.nms_kernel // 2) * self._bev_cell_cm:.0f} cm) | "
                  f"radius={self.nms_radius_cm:.0f} cm "
                  f"({self.nms_radius_cm / self._bev_cell_cm:.1f} cells) | "
                  f"iou={self.nms_iou:.2f}")
        return self._bev_cell_cm

    def _nms_kwargs(self, item):
        cell = self._bev_cell_size_cm(item)
        # (1) nms_score_threshold == conf_threshold COUPLED suppression to
        #     the reporting threshold, so a confidence sweep was not a sweep
        #     of one fixed model. Use a low fixed value instead.
        # (2) rotated-IoU NMS is only defined when the box heads are
        #     actually supervised. In a 2D-only run they emit the init
        #     prior, so the stage is arbitrary -> disable it there.
        return dict(
            nms_kernel=self.nms_kernel,
            nms_radius=(self.nms_radius_cm / cell
                        if self.nms_radius_cm > 0 else 0.0),
            nms_iou=(self.nms_iou if self.has_3d_gt else 0.0),
            nms_cell_size_cm=cell,
            nms_score_threshold=float(self.nms_score_threshold),
        )

    def _make_tracker(self):
        cell = self._bev_cell_cm or self._grid_cell_cm or 10.0
        if self._bev_cell_cm is None:
            print(f"[TRACK] WARNING: tracker built before the first decode; "
                  f"falling back to cell={cell:.1f} cm "
                  f"(assoc={self.track_assoc_dist_cm / cell:.1f} cells, "
                  f"dup={self.track_dup_dist_cm / cell:.1f} cells).")
        # 3D attribute smoothing only makes sense when the yaw/size/posture
        # heads are actually supervised; in a 2D-only run they emit the init
        # prior, so smoothing a constant is pointless (and misleading).
        use_3d = bool(self.track_use_3d_attrs and self.has_3d_gt)
        if self.track_use_3d_attrs and not self.has_3d_gt:
            print("[TRACK3D] track_use_3d_attrs=True but has_3d_gt=False "
                  "-> 3D attribute smoothing DISABLED for this run.")
        # IoU association needs per-detection 3D attrs
        use_iou = bool(self.track_assoc_use_iou and use_3d)
        if self.track_assoc_use_iou and not use_3d:
            print("[ASSOC-IOU] track_assoc_use_iou=True but the 3D attribute "
                  "stream is off (track_use_3d_attrs=False or has_3d_gt=False) "
                  "-> IoU association DISABLED for this run.")
        self._assoc_iou_active = use_iou
        # Lying-freeze needs the fused posture, i.e. the 3D attribute
        # stream. Force it off (loudly) when that stream is unavailable.
        freeze = bool(self.track_lying_velocity_freeze and use_3d)
        if self.track_lying_velocity_freeze and not freeze:
            print("[LYING-FREEZE] track_lying_velocity_freeze=True but the "
                  "3D attribute stream is off (track_use_3d_attrs=False or "
                  "has_3d_gt=False) -> lying-freeze DISABLED for this run.")
        common = dict(
            conf_thres=self.conf_threshold,
            track_buffer=self.track_buffer,
            dup_dist=self.track_dup_dist_cm / cell,
            assoc_dist=self.track_assoc_dist_cm / cell,
            use_3d_attrs=use_3d,
            attr_cfg=self.track_attr_cfg,
            track_writeback_to_detections=self.track_writeback_to_detections,
            assoc_iou_weight=(self.track_assoc_iou_weight if use_iou else 0.0),
            cell_cm=cell,
            freeze_when_lying=freeze,
        )
        return JDETracker(**common)

    # ==================================================================
    # canonical OBB export (additive; no metric depends on it)
    # ==================================================================
    @torch.no_grad()
    def _accumulate_obb_export(self, item, output):
        """Decode one test batch into canonical OBB records.

        Schema is chosen from the Phase-2 ``has_3d_gt`` flag:
            has_3d_gt=True  -> utils.obb.OBB_FIELDS_3D (11 cols)
            has_3d_gt=False -> utils.obb.OBB_FIELDS_2D (4 cols, POINTS ONLY)

        A 2D-only run can never emit yaw/size/posture: we do not even
        pass the 3D heads to ``decode.decoder`` in that branch.
        """
        # Defensive: has_3d_gt is a *dataset* property; also require that
        # the model actually produced the per-object attribute head.
        attr_fn = self._attr_fn_3d(output) if self.has_3d_gt else None
        has3d = bool(self.has_3d_gt) and attr_fn is not None
        if self.has_3d_gt and not has3d and not self._obb_export_warned:
            print("[OBB] WARNING: has_3d_gt=True but the model emitted no "
                  "attr_feat/attr3d_head -> falling back to a POINTS-ONLY "
                  "OBB export.")
            self._obb_export_warned = True

        if self._obb_export_has3d is None:
            self._obb_export_has3d = has3d
        elif self._obb_export_has3d != has3d:
            # Schema must stay constant for the whole file.
            if not self._obb_export_warned:
                print("[OBB] WARNING: OBB export schema changed mid-epoch; "
                      "keeping the first schema and skipping this batch.")
                self._obb_export_warned = True
            return

        records = obb.decode_obb(
            output, item, self.vox_util,
            conf_threshold=self.conf_threshold,  # same semantics as test_step
            has_3d_gt=has3d,
            attr_fn=attr_fn,
            Y=self.Y, Z=self.Z, X=self.X,
            size_scale_cm=self.size_scale_cm,
            max_detections=self.max_detections,
            coord_space='world_cm',  # centimetres
            use_global_frame_id=True,  # seq*1e6 + frame
            **self._nms_kwargs(item),
        )
        if records:
            self.obb_export_list.extend(records)

    def _write_obb_export(self, log_dir):
        """Write the canonical OBB export produced during test_step.

        Creates a NEW file only:
            obb_detections_full.txt    (has_3d_gt=True,  11 cols)
            obb_detections_points.txt  (has_3d_gt=False,  4 cols)
        Never touches moda_pred_3d.txt / mota_pred_3d.txt /
        gt_OBB_*.txt / pred_OBB_*.txt.
        """
        has3d = self._obb_export_has3d
        if has3d is None:
            return

        fields = obb.obb_fields(has3d)
        n_fields = len(fields)
        fname = ('obb_detections_full.txt' if has3d
                 else 'obb_detections_points.txt')
        out_path = osp.join(log_dir, fname)

        arr = np.asarray(
            self.obb_export_list, dtype=np.float64
        ).reshape(-1, n_fields)

        if has3d:
            fmt = ['%d', '%.4f', '%.4f', '%.4f', '%.6f',
                   '%.4f', '%.4f', '%.4f', '%d', '%.6f', '%.6f']
            units = ("frame_id = sequence_num*1000000 + frame; "
                     "x_cm,y_cm,z_base_cm,length_cm,width_cm,height_cm in "
                     "CENTIMETRES (world frame); yaw_rad in RADIANS "
                     "[-pi,pi]; posture_class 1=standing 0=lying; "
                     "posture_prob,score in [0,1]")
        else:
            fmt = ['%d', '%.4f', '%.4f', '%.6f']
            units = ("frame_id = sequence_num*1000000 + frame; "
                     "x_cm,y_cm in CENTIMETRES (world frame); "
                     "score in [0,1]. POINTS ONLY -- this run has NO 3D "
                     "ground truth, so no yaw/size/posture is exported.")

        header = (f"{' '.join(fields)}\n"
                  f"{units}\n"
                  f"conf_threshold={self.conf_threshold} "
                  f"max_detections={self.max_detections} "
                  f"has_3d_gt={bool(self.has_3d_gt)}\n"
                  f"unlabeled_sequences (inference-only, excluded from "
                  f"all metrics): {self._test_unlabeled_seqs or 'none'}")
        np.savetxt(out_path, arr, fmt=fmt, header=header, comments='# ')

        print("\n" + "-" * 80)
        print("CANONICAL OBB EXPORT (utils.obb.decode_obb)")
        print("-" * 80)
        print(f"  Schema : "
              f"{'3D oriented boxes' if has3d else 'POINTS ONLY (2D-only run)'}")
        print(f"  Rows   : {arr.shape[0]}   Cols: {n_fields}")
        print(f"  File   -> {out_path}")
        print("  cols: " + ' '.join(fields))
        print("  units: " + units)
        print("-" * 80)

    # ==================================================================
    # Oriented-bounding-box (OBB) evaluation
    # ==================================================================
    _OBB_DOC = (
        "Box convention : (cx, cy, length, width, yaw); length along the "
        "box's local +x,\n"
        "                 width along local +y.\n"
        "Units          : world CENTIMETRES, BEV (ground) plane.\n"
        "Angle          : yaw in RADIANS, CCW positive.\n"
        "Association    : rotated BEV IoU + OPTIMAL (Hungarian) assignment "
        "on cost = 1-IoU,\n"
        "                 pairs with IoU < threshold forbidden. NOT greedy.\n"
        "MODP           : mean IoU over matched (TP) pairs, in percent.\n"
        "MOTP           : mean IoU over matched pairs, in percent "
        "(= 100*(1-motmetrics motp)).\n"
        "Track boxes    : sampled from the dense yaw/size/posture heads at "
        "each track's OWN\n"
        "                 BEV cell (no snapping, no radius gate), so every "
        "track in the point\n"
        "                 MOTA file also appears here.\n"
        "CAVEAT         : rotated-rectangle IoU is INVARIANT to a 180 deg "
        "yaw flip; heading\n"
        "                 correctness is covered by the pose3d/yaw_* "
        "metrics above.\n"
        "CAVEAT         : the point-based MODA/MOTA above use a EUCLIDEAN "
        "gate, these use an\n"
        "                 IoU gate -> the two families of numbers are NOT "
        "directly comparable."
    )

    def _evaluate_obb(self, log_dir):
        """OBB CLEAR-MOD + CLEAR-MOT, overall and per sequence.

        Reads only the in-memory OBB lists (the exact contents of
        gt_OBB_*.txt / pred_OBB_*.txt). Touches no point-based
        accumulator, file or logged key.
        """
        import pandas as pd
        from evaluation import obb_metrics as om

        print("\n" + "-" * 80)
        print("ORIENTED-BOUNDING-BOX (OBB) EVALUATION")
        print("-" * 80)
        for line in self._OBB_DOC.splitlines():
            print("  " + line)

        if not self.obb_gt_moda:
            print("\n  No 3D ground truth was accumulated in this test run")
            print("  (dataset has no 3D_annotations/, or the model emitted "
                  "no 3D heads).")
            print("  -> OBB evaluation SKIPPED. All point-based metrics "
                  "above are unaffected.")
            print("-" * 80)
            return

        # track-box coverage
        n_pt_tracks = len(self.mota_pred_list)
        n_obb_tracks = len(self.obb_pred_mota)
        print(f"\n  Emitted tracks (point MOTA file) : {n_pt_tracks}")
        print(f"  Emitted tracks (OBB  MOTA file)  : {n_obb_tracks}")
        print(f"  Tracks WITHOUT an OBB            : "
              f"{self.obb_tracks_missing_box} "
              f"(cell outside the BEV grid)")
        self.log('obb_track/tracks_total', float(self.obb_tracks_total))
        self.log('obb_track/tracks_without_box',
                 float(self.obb_tracks_missing_box))

        # detection
        DET_COLS = (1, 2, 4, 5, 3)  # -> x, y, length, width, yaw
        gt_det = np.asarray(self.obb_gt_moda, dtype=np.float64).reshape(-1, 6)
        pred_det = (np.asarray(self.obb_pred_moda,
                               dtype=np.float64).reshape(-1, 6)
                    if self.obb_pred_moda else np.zeros((0, 6)))
        gt_by_f = om.rows_to_frame_dict(gt_det, 0, DET_COLS)
        pred_by_f = om.rows_to_frame_dict(pred_det, 0, DET_COLS)
        seqs = sorted({k // 1_000_000 for k in gt_by_f})

        det_rows = []
        for thr in self.obb_iou_thresholds:
            key = f'obb_detect/iou{thr:.2f}'
            res = om.obb_mod_metrics(gt_by_f, pred_by_f, thr)
            print()
            om.print_mod_report(res,
                                title='OBB DETECTION (CLEAR-MOD) -- OVERALL')
            for k in ('recall', 'precision', 'moda', 'modp'):
                self.log(f'{key}/overall_{k}', res[k])
            self.log(f'{key}/overall_tp', float(res['tp']))
            self.log(f'{key}/overall_fp', float(res['fp']))
            self.log(f'{key}/overall_fn', float(res['fn']))
            det_rows.append(dict(iou_threshold=thr, sequence='OVERALL',
                                 **{k: res[k] for k in
                                    ('n_frames', 'n_gt', 'n_pred', 'tp',
                                     'fp', 'fn', 'recall', 'precision',
                                     'moda', 'modp')}))

            print(f"{'Seq':<6}{'GT':>8}{'TP':>8}{'FP':>8}{'FN':>8}"
                  f"{'Recall':>10}{'Prec':>10}{'MODA':>10}{'MODP':>10}")
            print("-" * 78)
            for seq in seqs:
                g = {k: v for k, v in gt_by_f.items()
                     if k // 1_000_000 == seq}
                p = {k: v for k, v in pred_by_f.items()
                     if k // 1_000_000 == seq}
                if not g:
                    print(f"{seq:<6}  no 3D GT -> skipped")
                    continue
                r = om.obb_mod_metrics(g, p, thr)
                print(f"{seq:<6}{r['n_gt']:>8}{r['tp']:>8}{r['fp']:>8}"
                      f"{r['fn']:>8}{r['recall']:>10.2f}"
                      f"{r['precision']:>10.2f}{r['moda']:>10.2f}"
                      f"{r['modp']:>10.2f}")
                for k in ('recall', 'precision', 'moda', 'modp'):
                    self.log(f'{key}/seq{seq}_{k}', r[k])
                det_rows.append(dict(iou_threshold=thr, sequence=seq,
                                     **{k: r[k] for k in
                                        ('n_frames', 'n_gt', 'n_pred', 'tp',
                                         'fp', 'fn', 'recall', 'precision',
                                         'moda', 'modp')}))
            print("-" * 78)

        pd.DataFrame(det_rows).to_csv(
            osp.join(log_dir, 'obb_detection_metrics_per_sequence.csv'),
            index=False)

        # OBB detection AFTER rotated-IoU NMS
        if self.obb_nms_iou > 0 and self.moda_pred_3d_list:
            p3 = np.asarray(self.moda_pred_3d_list, np.float64).reshape(-1, 9)
            kept_rows, n_in, n_out = [], 0, 0
            for f in np.unique(p3[:, 0]):
                rows = p3[p3[:, 0] == f]
                bx = rows[:, [1, 2, 4, 5, 3]]  # x,y,L,W,yaw
                n_in += len(rows)
                for i in om.nms_obb(bx, rows[:, 8], self.obb_nms_iou):
                    kept_rows.append(rows[i]);
                    n_out += 1
            kept = np.asarray(kept_rows, np.float64).reshape(-1, 9)
            nms_by_f = om.rows_to_frame_dict(kept, 0, (1, 2, 4, 5, 3))
            np.savetxt(osp.join(log_dir, 'pred_OBB_moda_nms.txt'),
                       kept[:, [0, 1, 2, 3, 4, 5]], '%f')
            print(f"\n  ROTATED-IoU NMS (IoU >= {self.obb_nms_iou:.2f}): "
                  f"{n_in} -> {n_out} predictions "
                  f"({n_in - n_out} duplicates removed, "
                  f"{100.0 * (n_in - n_out) / max(n_in, 1):.2f}%)")
            self.log('obb_detect_nms/removed_fraction',
                     float((n_in - n_out) / max(n_in, 1)))
            for thr in self.obb_iou_thresholds:
                r = om.obb_mod_metrics(gt_by_f, nms_by_f, thr)
                print()
                om.print_mod_report(
                    r, title=f'OBB DETECTION AFTER NMS -- OVERALL')
                k = f'obb_detect_nms/iou{thr:.2f}'
                for f_ in ('recall', 'precision', 'moda', 'modp'):
                    self.log(f'{k}/overall_{f_}', r[f_])
                det_rows.append(dict(iou_threshold=thr, sequence='OVERALL_NMS',
                                     **{f_: r[f_] for f_ in
                                        ('n_frames', 'n_gt', 'n_pred', 'tp',
                                         'fp', 'fn', 'recall', 'precision',
                                         'moda', 'modp')}))
            pd.DataFrame(det_rows).to_csv(
                osp.join(log_dir, 'obb_detection_metrics_per_sequence.csv'),
                index=False)

        # tracking
        TRK_COLS = (3, 4, 6, 7, 5)  # -> x, y, length, width, yaw
        gt_trk = (np.asarray(self.obb_gt_mota,
                             dtype=np.float64).reshape(-1, 8)
                  if self.obb_gt_mota else np.zeros((0, 8)))
        pred_trk = (np.asarray(self.obb_pred_mota,
                               dtype=np.float64).reshape(-1, 8)
                    if self.obb_pred_mota else np.zeros((0, 8)))
        gt_by_sf = om.rows_to_seq_frame_dict(gt_trk, 0, 1, 2, TRK_COLS)
        pred_by_sf = om.rows_to_seq_frame_dict(pred_trk, 0, 1, 2, TRK_COLS)

        trk_rows = []
        for thr in self.obb_iou_thresholds:
            summary = om.obb_mot_metrics(gt_by_sf, pred_by_sf, thr)
            if summary is None:
                print("\n  OBB CLEAR-MOT skipped (motmetrics unavailable "
                      "or no 3D GT tracks).")
                break
            print()
            om.print_mot_report(summary, thr)
            key = f'obb_track/iou{thr:.2f}'
            for name, row in summary.iterrows():
                tag = ('overall' if str(name) == 'OVERALL' else str(name))
                motp = row['motp']
                motp_iou = (100.0 * (1.0 - motp)) if motp == motp else 0.0
                self.log(f'{key}/{tag}_mota', 100.0 * row['mota'])
                self.log(f'{key}/{tag}_motp_iou', motp_iou)
                self.log(f'{key}/{tag}_idf1', 100.0 * row['idf1'])
                self.log(f'{key}/{tag}_num_switches',
                         float(row['num_switches']))
                self.log(f'{key}/{tag}_mostly_tracked',
                         float(row['mostly_tracked']))
                self.log(f'{key}/{tag}_mostly_lost',
                         float(row['mostly_lost']))
                self.log(f'{key}/{tag}_num_fragmentations',
                         float(row['num_fragmentations']))
                trk_rows.append(dict(
                    iou_threshold=thr, sequence=tag,
                    mota=100.0 * row['mota'], motp_iou=motp_iou,
                    idf1=100.0 * row['idf1'],
                    num_switches=int(row['num_switches']),
                    mostly_tracked=int(row['mostly_tracked']),
                    mostly_lost=int(row['mostly_lost']),
                    num_fragmentations=int(row['num_fragmentations']),
                    num_unique_objects=int(row['num_unique_objects']),
                ))

        if trk_rows:
            pd.DataFrame(trk_rows).to_csv(
                osp.join(log_dir, 'obb_tracking_metrics_per_sequence.csv'),
                index=False)


    def test_step(self, batch, batch_idx):
        item, target = batch

        start_time = time.perf_counter()
        output = self(item)

        center_e = output['instance_center'].float()
        offset_e = output['instance_offset'].float()

        with torch.autocast(device_type=self.device.type, enabled=False):
            decoded = decode.decoder(
                center_e.sigmoid(), offset_e, None,
                K=self.max_detections,
                attr_fn=(self._attr_fn_3d(output)
                         if self.has_3d_gt else None),
                size_scale=self.size_scale_cm,
                **self._nms_kwargs(item),
            )
        if len(decoded) == 5:
            xy_e, xy_prev_e, scores_e, classes_e, extra_e = decoded
        else:
            xy_e, xy_prev_e, scores_e, classes_e = decoded
            extra_e = {}

        # mem-grid -> world-grid (shared helper, identical arithmetic)
        ref_xy = obb.mem_to_ref_xy(
            self.vox_util, xy_e, self.Y, self.Z, self.X
        )
        ref_xy_prev = obb.mem_to_ref_xy(
            self.vox_util, xy_prev_e, self.Y, self.Z, self.X
        )

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

        # detection metrics (BEV grid coords)
        for frame, grid_gt, xy, score, seq_num in zip(
                item['frame'], item['grid_gt'], ref_xy, scores_e,
                item['sequence_num'],
        ):
            frame_val = int(frame.item())
            seq_val = int(seq_num.item())
            global_frame = seq_val * 1_000_000 + frame_val

            valid = score > self.conf_threshold
            if not self._seq_has_gt.get(seq_val, True):
                self.moda_pred_unlabeled_list.extend([
                    [global_frame, x.item(), y.item()]
                    for x, y in xy[valid]
                ])
                continue
            self.moda_gt_list.extend([
                [global_frame, x.item(), y.item()]
                for x, y, _ in grid_gt[grid_gt.sum(1) != 0]
            ])
            self.moda_pred_list.extend([
                [global_frame, x.item(), y.item()]
                for x, y in xy[valid]
            ])

        # numpy views of the 3D predictions
        yaw_np = (extra_e['yaw_angle'].float().cpu().numpy()
                  if 'yaw_angle' in extra_e else None)
        dim_np = (extra_e['dimensions'].float().cpu().numpy()
                  if 'dimensions' in extra_e else None)
        post_np = (extra_e['posture_class'].long().cpu().numpy()
                   if 'posture_class' in extra_e else None)
        pprob_np = (extra_e['posture_prob'].float().cpu().numpy()
                    if 'posture_prob' in extra_e else None)
        has3d_pred = (yaw_np is not None and dim_np is not None
                      and post_np is not None)

        g3d_all = item.get('grid_gt_3d', None)
        has3d_gt = (g3d_all is not None and g3d_all.shape[-1] >= 11)

        Wm = item['worldcoord_from_worldgrid'].float().cpu()
        if Wm.dim() == 2:
            Wm = Wm.unsqueeze(0).expand(batch_size, -1, -1)

        if self._grid_cell_cm is None:
            # W maps world-grid cells -> world cm; 10.0 for mmCows.
            self._grid_cell_cm = abs(float(Wm[0, 0, 0]))

        scores_cpu = scores_e.detach().cpu()

        # BEV tracking + 3D property evaluation
        for b_idx, (seq_num, frame, grid_gt, bev_det, bev_prev,
                    score) in enumerate(zip(
                item['sequence_num'], item['frame'], item['grid_gt'],
                ref_xy.cpu(), ref_xy_prev.cpu(), scores_e.cpu(),
        )):
            frame_val = int(frame.item())
            seq_val = int(seq_num.item())
            global_frame = seq_val * 1_000_000 + frame_val
            # Per-sequence GT routing: True (default) reproduces the old
            # behaviour for datasets without the has_gt attribute.
            seq_has_gt = self._seq_has_gt.get(seq_val, True)
            # 3D metric streams accept rows only from sequences whose GT
            # can JUDGE 3D predictions (annotated AND has_3d).
            seq_has_3d_gt = (seq_has_gt
                             and self._seq_has_3d.get(seq_val, False))

            if seq_val != self._current_test_seq:
                # new sequence: start with a fresh tracker
                self.test_tracker = self._make_tracker()
                self._current_test_seq = seq_val

            # hand the per-detection 3D attributes to the tracker so
            # each track can maintain its own running yaw/size/posture.
            det_attrs = None
            if (self.track_use_3d_attrs and yaw_np is not None
                    and dim_np is not None):
                det_attrs = {
                    'yaw': yaw_np[b_idx],
                    'dims': dim_np[b_idx],
                    'posture_prob': (pprob_np[b_idx]
                                     if pprob_np is not None else None),
                }
            # tracker-only timing
            _t0 = time.perf_counter()
            output_stracks = self.test_tracker.update(
                bev_det, bev_prev, score, attrs=det_attrs)
            self._tracker_times.append((time.perf_counter() - _t0) * 1000)

            # BEV MOT accumulation. GT rows and metric-bound predictions
            # only for annotated sequences
            if seq_has_gt:
                self.mota_gt_list.extend([
                    [seq_val, frame_val, i.item(),
                     -1, -1, -1, -1, 1,
                     x.item(), y.item(), -1]
                    for x, y, i in grid_gt[grid_gt.sum(1) != 0]
                ])
            mota_pred_target = (self.mota_pred_list if seq_has_gt
                                else self.mota_pred_unlabeled_list)
            mota_pred_target.extend([
                [seq_val, frame_val, s.track_id,
                 -1, -1, -1, -1, float(s.score)]
                + s.xy.tolist() + [-1]
                for s in output_stracks
            ])

            if not (has3d_pred and has3d_gt):
                continue

            # grid → world (cm) via the shared helper
            W3 = Wm[b_idx].numpy()  # (3, 3) worldcoord_from_worldgrid

            keep = (score > self.conf_threshold).numpy()
            det_grid = bev_det.float().cpu().numpy()  # (K, 2) grid
            p_grid = det_grid[keep]
            p_world = obb.worldgrid_to_worldcm(p_grid, W3)  # (P, 2) cm
            p_yaw = yaw_np[b_idx][keep]
            p_dim = dim_np[b_idx][keep]
            p_post = post_np[b_idx][keep]
            p_score = scores_cpu[b_idx].float().numpy()[keep]

            g = g3d_all[b_idx].float().cpu().numpy()
            g = g[g[:, 5] > 0]                            # real entries
            gt_world = g[:, 2:4]
            gt_yaw, gt_dim, gt_post = g[:, 6], g[:, 7:10], g[:, 10]


            # propagate each track's SMOOTHED box back onto its own
            # detection, so obb_detect/* and pose3d/* also benefit.
            if self.track_use_3d_attrs and self.track_writeback_to_detections:
                # det_index is an index into the TRACKER's masked arrays
                # (score > conf_threshold - 0.1, see JDETracker.update),
                # while p_yaw/p_dim/p_post are masked by `keep`
                # (score > conf_threshold). Map:
                #   tracker index  ->  Top-K index  ->  position in p_*.
                tr_topk = np.nonzero(
                    score.float().numpy() > self.conf_threshold - 0.1
                )[0]  # tracker det -> Top-K
                pos_of_topk = {int(k): j
                               for j, k in enumerate(np.nonzero(keep)[0])}
                for s in output_stracks:
                    di = getattr(s, 'det_index', None)
                    if di is None or not getattr(s, 'has_3d', False):
                        continue
                    if di >= len(tr_topk):
                        continue
                    j = pos_of_topk.get(int(tr_topk[di]))
                    if j is None:
                        continue  # detection below conf_threshold: not in p_*
                    p_yaw[j] = float(s.yaw3d)
                    p_dim[j] = np.asarray(s.dims3d, dtype=p_dim.dtype)
                    p_post[j] = int(s.posture3d)
            # enriched detection predictions (world cm)
            det3d_target = (self.moda_pred_3d_list if seq_has_3d_gt
                            else self.moda_pred_3d_unlabeled_list)
            for i in range(len(p_world)):
                det3d_target.append([
                    global_frame, p_world[i, 0], p_world[i, 1],
                    p_yaw[i], p_dim[i, 0], p_dim[i, 1], p_dim[i, 2],
                    float(p_post[i]), float(p_score[i]),
                ])
                if seq_has_3d_gt:
                    self.obb_pred_moda.append([
                        global_frame, p_world[i, 0], p_world[i, 1],
                        p_yaw[i], p_dim[i, 0], p_dim[i, 1],
                    ])

            for k in range(len(g)):
                self.obb_gt_moda.append([
                    global_frame, gt_world[k, 0], gt_world[k, 1],
                    gt_yaw[k], gt_dim[k, 0], gt_dim[k, 1],
                ])
                self.obb_gt_mota.append([
                    seq_val, frame_val, int(g[k, 5]),
                    gt_world[k, 0], gt_world[k, 1],
                    gt_yaw[k], gt_dim[k, 0], gt_dim[k, 1],
                ])
                self.mota_gt_3d_list.append([
                    seq_val, frame_val, int(g[k, 5]),
                    gt_world[k, 0], gt_world[k, 1],
                    gt_yaw[k], gt_dim[k, 0], gt_dim[k, 1], gt_dim[k, 2],
                    float(gt_post[k]),
                ])

            # greedy score-ordered association for 3D metrics
            order = np.argsort(-p_score)
            matches = self._greedy_match(
                p_world[order], gt_world, self.eval_match_dist_cm
            )
            for pi_o, gi in matches:
                pi = int(order[pi_o])
                self.posture_total += 1
                if int(p_post[pi]) == int(round(float(gt_post[gi]))):
                    self.posture_correct += 1
                self.yaw_errors.append(
                    float(self._wrap_half_pi(p_yaw[pi] - gt_yaw[gi]))
                )
                self.dim_errors.append(
                    np.abs(p_dim[pi] - gt_dim[gi]).tolist()
                )

            # EVERY emitted track gets a well-defined OBB
            if output_stracks:
                t_ref = np.stack([
                    np.asarray(s.xy, dtype=np.float32).reshape(2)
                    for s in output_stracks
                ], axis=0)  # (T, 2) grid
                t_ref3 = torch.from_numpy(np.concatenate(
                    [t_ref, np.zeros((len(t_ref), 1), np.float32)], axis=1
                )).unsqueeze(0)  # (1, T, 3)
                t_mem = self.vox_util.Ref2Mem(
                    t_ref3, self.Y, self.Z, self.X
                )[0, :, :2].cpu().numpy()  # (T, 2) mem

                # Routed targets: metric-bound lists only for sequences
                # with 3D GT; unlabelled/2D sequences export separately.
                trk3d_target = (self.mota_pred_3d_list if seq_has_3d_gt
                                else self.mota_pred_3d_unlabeled_list)
                trkobb_target = (self.obb_pred_mota if seq_has_3d_gt
                                 else self.obb_pred_mota_unlabeled)

                for ti, s in enumerate(output_stracks):
                    # obb_tracks_total / _missing_box are the parity
                    # diagnostic for mota_pred.txt vs pred_OBB_mota.txt —
                    # count only tracks that go into those (metric) files.
                    if seq_has_3d_gt:
                        self.obb_tracks_total += 1

                    # this track's OWN smoothed 3D state
                    if self.track_use_3d_attrs and getattr(s, 'has_3d', False):
                        yv = float(s.yaw3d)
                        dv = np.asarray(s.dims3d, dtype=np.float64)
                        pv = int(s.posture3d)
                        self.track3d_smoothed += 1
                        if getattr(s.attr3d, 'flipped_last', False):
                            self.track3d_yaw_flips += 1
                    else:
                        # per-object query at the track's OWN sub-pixel
                        # BEV position (no rounding, no dense map).
                        mx = float(t_mem[ti, 0])
                        my = float(t_mem[ti, 1])
                        if not (0 <= mx < self.X and 0 <= my < self.Y):
                            if seq_has_3d_gt:
                                self.obb_tracks_missing_box += 1
                            continue
                        yv, dv, pv = self._sample_query_3d(
                            output['attr_feat'][b_idx:b_idx + 1], mx, my)
                        self.track3d_fallback += 1

                    w_xy = obb.worldgrid_to_worldcm(
                        t_ref[ti].astype(np.float64), W3
                    )
                    trk3d_target.append([
                        seq_val, frame_val, s.track_id, float(s.score),
                        w_xy[0], w_xy[1], yv, dv[0], dv[1], dv[2], pv,
                    ])
                    trkobb_target.append([
                        seq_val, frame_val, s.track_id,
                        w_xy[0], w_xy[1], yv, dv[0], dv[1],
                    ])

        # canonical OBB export
        self._accumulate_obb_export(item, output)


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
        n_unl = len(self._test_unlabeled_seqs)
        if n_unl:
            print(f"  Test set: {len(self._seq_has_gt) - n_unl} annotated "
                  f"+ {n_unl} inference-only sequence(s) "
                  f"{self._test_unlabeled_seqs}")
            print("  All metrics below are computed on the ANNOTATED "
                  "sequences only.")
        print("=" * 80)

        # ══════════════════════════════════════════════════
        # OVERALL DETECTION METRICS
        # ══════════════════════════════════════════════════
        pred_path = osp.join(log_dir, 'moda_pred.txt')
        gt_path = osp.join(log_dir, 'moda_gt.txt')
        np.savetxt(pred_path,
                   np.array(self.moda_pred_list).reshape(-1, 3), '%f')
        np.savetxt(gt_path,
                   np.array(self.moda_gt_list).reshape(-1, 3), '%d')
        # Single source for the evaluation association gate. ...
        eval_gate_cm = 100.0
        if self._grid_cell_cm is not None:
            eval_cell_cm = float(self._grid_cell_cm)
        else:
            eval_cell_cm = 100.0 if self.X == 150 else 2.5
            print(f"[EVAL-CHECK] WARNING: worldcoord_from_worldgrid was "
                  f"never seen; GUESSING grid_cell_cm={eval_cell_cm} "
                  f"from X={self.X}. For mmCows the correct value is 10.0.")
        td_cells = eval_gate_cm / eval_cell_cm
        if self.moda_gt_list:
            recall, precision, moda, modp = modMetricsCalculator(
                osp.abspath(pred_path), osp.abspath(gt_path),
                td_cells=td_cells,
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
        else:
            print("\n" + "-" * 80)
            print("OVERALL DETECTION METRICS")
            print("-" * 80)
            print("  SKIPPED — no annotated sequence in this test run "
                  "(moda_gt.txt is empty). See the *_unlabeled.txt "
                  "exports below.")

        # ══════════════════════════════════════════════════
        # PER-SEQUENCE DETECTION METRICS
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
                        td_cells=td_cells,
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
        # OVERALL TRACKING METRICS
        # ══════════════════════════════════════════════════
        # metres per world-grid cell, from the same cell size used above.
        scale = eval_cell_cm / 100.0
        mot_gate_m = eval_gate_cm / 100.0
        print(f"[EVAL-CHECK] grid_cell_cm={eval_cell_cm:.2f}  "
              f"td_cells={td_cells:.2f}  mot_scale={scale:.3f}  "
              f"mot_gate_m={mot_gate_m:.3f}  "
              f"gate_cm={eval_gate_cm:.1f}")
        pred_path = osp.join(log_dir, 'mota_pred.txt')
        gt_path = osp.join(log_dir, 'mota_gt.txt')
        np.savetxt(
            pred_path, np.array(self.mota_pred_list).reshape(-1, 11),
            '%f', delimiter=',',
        )
        np.savetxt(
            gt_path, np.array(self.mota_gt_list).reshape(-1, 11),
            '%f', delimiter=',',
        )
        if self.mota_gt_list:
            summary = mot_metrics(
                osp.abspath(pred_path), osp.abspath(gt_path),
                scale=scale,
                max_dist_m=mot_gate_m,
            )
            summary = summary.loc['OVERALL']
        else:
            summary = None
            print("\n" + "-" * 80)
            print("OVERALL TRACKING METRICS")
            print("-" * 80)
            print("  SKIPPED — no annotated sequence in this test run "
                  "(mota_gt.txt is empty). See mota_pred_unlabeled.txt.")

        if summary is not None:
            mota_pct = 100.0 * float(summary['mota'])
            idf1_pct = 100.0 * float(summary['idf1'])
            idp_pct = 100.0 * float(summary['idp'])
            idr_pct = 100.0 * float(summary['idr'])
            recall_pct = 100.0 * float(summary['recall'])
            precision_pct = 100.0 * float(summary['precision'])

            # motmetrics MOTP here is mean association DISTANCE in metres,
            # not a percentage. Also report a gate-normalized score.
            motp_distance_m = float(summary['motp'])
            motp_gate_pct = 100.0 * max(
                0.0, 1.0 - motp_distance_m / float(mot_gate_m))

            mt_count = int(summary['mostly_tracked'])
            ml_count = int(summary['mostly_lost'])
            n_objects = max(1, int(summary['num_unique_objects']))

            print("\n" + "-" * 80)
            print("OVERALL TRACKING METRICS")
            print("-" * 80)
            print(f"  MOTA:       {mota_pct:.2f}%")
            print(f"  MOTP dist:  {motp_distance_m:.3f} m "
                  f"({motp_gate_pct:.2f}% of the {mot_gate_m:.2f} m gate)")
            print(f"  IDF1:       {idf1_pct:.2f}%")
            print(f"  IDP / IDR:  {idp_pct:.2f}% / {idr_pct:.2f}%")
            print(f"  Recall:     {recall_pct:.2f}%")
            print(f"  Precision:  {precision_pct:.2f}%")
            print(f"  MT / ML:    {mt_count} / {ml_count} tracks "
                  f"({100.0 * mt_count / n_objects:.2f}% / "
                  f"{100.0 * ml_count / n_objects:.2f}%)")

            # Explicit logging avoids the old generic unit conversion...
            self.log('track/mota', mota_pct)
            self.log('track/idf1', idf1_pct)
            self.log('track/idp', idp_pct)
            self.log('track/idr', idr_pct)
            self.log('track/recall', recall_pct)
            self.log('track/precision', precision_pct)
            self.log('track/motp', motp_gate_pct)
            self.log('track/motp_distance_m', motp_distance_m)
            self.log('track/mostly_tracked', float(mt_count))
            self.log('track/mostly_lost', float(ml_count))
            self.log('track/mostly_tracked_pct',
                     100.0 * mt_count / n_objects)
            self.log('track/mostly_lost_pct',
                     100.0 * ml_count / n_objects)

            for key in ('num_frames', 'num_matches', 'num_switches',
                        'num_fragmentations', 'num_false_positives',
                        'num_misses', 'num_detections', 'num_objects',
                        'num_predictions', 'num_unique_objects',
                        'num_transfer', 'num_migrate', 'num_ascend'):
                if key in summary:
                    self.log(f'track/{key}', float(summary[key]))

        # Per-sequence BEV tracking debug files
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
        # UNLABELED-SEQUENCE EXPORTS (inference-only)
        # ══════════════════════════════════════════════════
        if self._test_unlabeled_seqs:
            import json as _json
            print("\n" + "-" * 80)
            print("UNLABELED SEQUENCES (inference-only — excluded from "
                  "every metric above)")
            print("-" * 80)
            print(f"  sequences: {self._test_unlabeled_seqs}")

            if self.moda_pred_unlabeled_list:
                p = osp.join(log_dir, 'moda_pred_unlabeled.txt')
                np.savetxt(p, np.array(
                    self.moda_pred_unlabeled_list).reshape(-1, 3), '%f')
                print(f"  detections (grid)      -> {p} "
                      f"({len(self.moda_pred_unlabeled_list)} rows)")
                print("    cols: frame_id x_grid y_grid "
                      "(frame_id = seq*1000000 + frame)")
            if self.mota_pred_unlabeled_list:
                p = osp.join(log_dir, 'mota_pred_unlabeled.txt')
                np.savetxt(p, np.array(
                    self.mota_pred_unlabeled_list).reshape(-1, 11),
                    '%f', delimiter=',')
                print(f"  point tracks (grid)    -> {p} "
                      f"({len(self.mota_pred_unlabeled_list)} rows)")
                print("    cols: seq frame id -1 -1 -1 -1 score "
                      "x_grid y_grid -1")
            if self.moda_pred_3d_unlabeled_list:
                p = osp.join(log_dir, 'moda_pred_3d_unlabeled.txt')
                np.savetxt(p, np.array(
                    self.moda_pred_3d_unlabeled_list).reshape(-1, 9), '%f')
                print(f"  detections + 3D (cm)   -> {p} "
                      f"({len(self.moda_pred_3d_unlabeled_list)} rows)")
                print("    cols: frame_id x_cm y_cm yaw L W H posture "
                      "score")
            if self.mota_pred_3d_unlabeled_list:
                p = osp.join(log_dir, 'mota_pred_3d_unlabeled.txt')
                np.savetxt(p, np.array(
                    self.mota_pred_3d_unlabeled_list).reshape(-1, 11),
                    '%f', delimiter=',')
                print(f"  tracks + 3D props (cm) -> {p} "
                      f"({len(self.mota_pred_3d_unlabeled_list)} rows)")
                print("    cols: seq frame id score x_cm y_cm yaw L W H "
                      "posture")
            if self.obb_pred_mota_unlabeled:
                p = osp.join(log_dir, 'obb_pred_mota_unlabeled.txt')
                np.savetxt(p, np.array(
                    self.obb_pred_mota_unlabeled).reshape(-1, 8),
                    '%f', delimiter=',')
                print(f"  OBB tracks (cm)        -> {p} "
                      f"({len(self.obb_pred_mota_unlabeled)} rows)")
                print("    cols: seq frame id x_cm y_cm yaw length width")

            routing = {
                'annotated_sequences': sorted(
                    s for s, g in self._seq_has_gt.items() if g),
                'unlabeled_sequences': list(self._test_unlabeled_seqs),
                'note': ("unlabeled sequences contribute NO rows to "
                         "moda_gt/moda_pred/mota_gt/mota_pred/OBB metric "
                         "files; their predictions live in the "
                         "*_unlabeled.txt files and in the canonical "
                         "obb_detections_*.txt export."),
            }
            rp = osp.join(log_dir, 'gt_routing.json')
            with open(rp, 'w') as f:
                _json.dump(routing, f, indent=1)
            print(f"  routing record         -> {rp}")

        # ══════════════════════════════════════════════════
        # 3D PROPERTY METRICS + ORIENTED-BOX (OBB) EXPORT
        # ══════════════════════════════════════════════════
        print("\n" + "-" * 80)
        print("3D PROPERTY METRICS (yaw / dimensions / posture)")
        print("-" * 80)

        if self.posture_total > 0:
            posture_acc = 100.0 * self.posture_correct / self.posture_total

            ye = np.abs(np.array(self.yaw_errors, dtype=np.float64))
            yaw_mae = float(np.degrees(ye.mean()))
            yaw_rmse = float(np.degrees(np.sqrt((ye ** 2).mean())))
            # 180°-ambiguity-tolerant variant (front/back flip ignored)
            yaw_mae_180 = float(
                np.degrees(np.minimum(ye, np.pi - ye).mean())
            )

            de = np.array(self.dim_errors, dtype=np.float64)
            l_mae, w_mae, h_mae = de.mean(axis=0).tolist()

            print(f"  Matched det/GT pairs : {self.posture_total} "
                  f"(gate = {self.eval_match_dist_cm:.0f} cm)")
            print(f"  Posture accuracy     : {posture_acc:.2f}%")
            print(f"  Yaw MAE              : {yaw_mae:.2f} deg")
            print(f"  Yaw RMSE             : {yaw_rmse:.2f} deg")
            print(f"  Yaw MAE (±180° amb.) : {yaw_mae_180:.2f} deg")
            print(f"  Length MAE           : {l_mae:.2f} cm")
            print(f"  Width  MAE           : {w_mae:.2f} cm")
            print(f"  Height MAE           : {h_mae:.2f} cm")

            self.log('pose3d/posture_accuracy', posture_acc)
            self.log('pose3d/yaw_mae_deg', yaw_mae)
            self.log('pose3d/yaw_rmse_deg', yaw_rmse)
            self.log('pose3d/yaw_mae_deg_180amb', yaw_mae_180)
            self.log('pose3d/length_mae_cm', l_mae)
            self.log('pose3d/width_mae_cm', w_mae)
            self.log('pose3d/height_mae_cm', h_mae)
            self.log('pose3d/num_matched', float(self.posture_total))
        else:
            print("  No matched prediction/GT pairs with 3D annotations "
                  "— 3D metrics skipped.")

        # enriched prediction files (world cm)
        if self.moda_pred_3d_list:
            p3d = osp.join(log_dir, 'moda_pred_3d.txt')
            np.savetxt(p3d, np.array(self.moda_pred_3d_list), '%f')
            print(f"\n  Detections + 3D props -> {p3d}")
            print("    cols: frame x_cm y_cm yaw L W H posture score")
        if self.mota_pred_3d_list:
            t3d = osp.join(log_dir, 'mota_pred_3d.txt')
            np.savetxt(t3d, np.array(self.mota_pred_3d_list),
                       '%f', delimiter=',')
            print(f"  Tracks + 3D props     -> {t3d}")
            print("    cols: seq frame id score x_cm y_cm yaw L W H posture")
        if self.mota_gt_3d_list:
            g3d = osp.join(log_dir, 'mota_gt_3d.txt')
            np.savetxt(g3d, np.array(self.mota_gt_3d_list),
                       '%f', delimiter=',')
            print(f"  GT tracks + 3D props  -> {g3d}")
            print("    cols: seq frame id x_cm y_cm yaw L W H posture")

        # OBB GT / pred files for oriented-box IoU eval
        if self.obb_gt_moda:
            g_obb = osp.join(log_dir, 'gt_OBB_moda.txt')
            p_obb = osp.join(log_dir, 'pred_OBB_moda.txt')
            np.savetxt(g_obb, np.array(self.obb_gt_moda), '%f')
            np.savetxt(
                p_obb,
                np.array(self.obb_pred_moda) if self.obb_pred_moda
                else np.empty((0, 6)),
                '%f',
            )
            print(f"  OBB detection GT      -> {g_obb}")
            print(f"  OBB detection pred    -> {p_obb}")
            print("    cols: frame x_cm y_cm yaw length width")
        if self.obb_gt_mota:
            g_obb_t = osp.join(log_dir, 'gt_OBB_mota.txt')
            p_obb_t = osp.join(log_dir, 'pred_OBB_mota.txt')
            np.savetxt(g_obb_t, np.array(self.obb_gt_mota),
                       '%f', delimiter=',')
            np.savetxt(
                p_obb_t,
                np.array(self.obb_pred_mota) if self.obb_pred_mota
                else np.empty((0, 8)),
                '%f', delimiter=',',
            )
            print(f" OBB tracking GT   -> {g_obb_t}")
            print(f" OBB tracking pred -> {p_obb_t}")
            print(" cols: seq frame id x_cm y_cm yaw length width")

            # ── 3D-aware tracking diagnostics (ablation bookkeeping) ──
        n_tr = self.track3d_smoothed + self.track3d_fallback
        print("\n 3D-AWARE TRACKING (track_use_3d_attrs = "
              f"{self.track_use_3d_attrs})")
        if self.track_use_3d_attrs:
            print(f"   size mode = {self.track_attr_cfg['size_mode']}, "
                  f"yaw mode = {self.track_attr_cfg['yaw_mode']}, "
                  f"flip_align = {self.track_attr_cfg['flip_align']}")
        print(f"   track-frames with SMOOTHED 3D attrs : "
              f"{self.track3d_smoothed}")
        print(f"   track-frames using per-frame sample : "
              f"{self.track3d_fallback}")
        print(f"   yaw 180 deg flips absorbed          : "
              f"{self.track3d_yaw_flips}")
        print(f"   lying velocity freeze               : "
              f"{self.track_lying_velocity_freeze}")
        tr = self.test_tracker
        if tr is not None and hasattr(tr, 'lying_diagnostics'):
            ld = tr.lying_diagnostics()
            print(f"   lying-frozen track-frames         : "
                  f"{ld['lying_frozen_trackframes']}")
            print(f"   lying velocity zeroings           : "
                  f"{ld['lying_velocity_zeroings']}")
        if self._tracker_times:
            tt = np.asarray(self._tracker_times)
            print(f"   tracker time per frame (ms)       : "
                  f"mean {tt.mean():.3f} | p95 "
                  f"{np.percentile(tt, 95):.3f} | max {tt.max():.3f}")
        if n_tr:
            self.log('track3d/smoothed_fraction',
                     float(self.track3d_smoothed) / float(n_tr))
        self.log('track3d/yaw_flips', float(self.track3d_yaw_flips))
        self.log('track3d/enabled', float(bool(self.track_use_3d_attrs)))
        print(f"   IoU association (track_assoc_use_iou) : "
              f"{self.track_assoc_use_iou} -> active = {self._assoc_iou_active}, "
              f"weight = {self.track_assoc_iou_weight if self._assoc_iou_active else 0.0}")
        self.log('track_assoc/iou_active', float(self._assoc_iou_active))
        self.log('track_assoc/iou_weight',
                 float(self.track_assoc_iou_weight
                       if self._assoc_iou_active else 0.0))

        print("-" * 80)

        # ══════════════════════════════════════════════════
        # CANONICAL OBB EXPORT
        # ══════════════════════════════════════════════════
        self._write_obb_export(log_dir)

        # ══════════════════════════════════════════════════
        # OBB CLEAR-MOD / CLEAR-MOT (rotated-IoU, Hungarian)
        # ══════════════════════════════════════════════════
        self._evaluate_obb(log_dir)

        # ══════════════════════════════════════════════════
        # INFERENCE TIMING
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
                tag = ('  (no GT — inference only)'
                       if int(seq) in set(self._test_unlabeled_seqs) else '')
                print(f"{int(seq):<6} {nf:<8} {sm:<10.2f} "
                      f"{smd:<12.2f} {ss:<10.2f} {sf:<8.2f}{tag}")
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
        # SAVE REFINED CALIBRATIONS (test snapshot)
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
    # Visual encoding of the OBB overlay (also written into the figure
    # title so the plot is self-documenting in TensorBoard):
    #
    #   ax1 (center_g)  ->  GROUND-TRUTH boxes, lime
    #   ax2 (center_e)  ->  PREDICTED boxes,    red
    #   solid outline   ->  posture = standing (1)
    #   dashed outline  ->  posture = lying    (0)
    #
    _OBB_GT_COLOR = 'lime'
    _OBB_PRED_COLOR = 'red'
    _OBB_LINEWIDTH = 1.4

    def _obb_footprint_mem(self, cx_ref, cy_ref, yaw, l_cm, w_cm, W):
        dev = 'cpu'
        l_cells = float(l_cm) / float(W[0, 0])
        w_cells = float(w_cm) / float(W[1, 1])

        def _t(v):
            return torch.tensor(float(v), dtype=torch.float32, device=dev)

        corners = obb.make_box_corners(
            _t(cx_ref), _t(cy_ref), _t(0.0),
            _t(l_cells), _t(w_cells), _t(0.0),
            _t(np.sin(yaw)), _t(np.cos(yaw)),
            dev,
        )[:4]  # (4, 3) bottom face

        mem = self.vox_util.Ref2Mem(
            corners.unsqueeze(0), self.Y, self.Z, self.X
        )[0, :, :2]
        return mem.detach().cpu().numpy()

    @torch.no_grad()
    def _draw_obb_overlay(self, ax_gt, ax_pred, item, output):
        """Overlay GT / predicted oriented boxes on the two heatmaps.

        Returns (n_gt_boxes, n_pred_boxes) actually drawn.
        """
        # Both panels display batch element [-1]; use the same one here.
        b = int(output['instance_center'].shape[0]) - 1

        W = item['worldcoord_from_worldgrid'].detach().float().cpu()
        if W.dim() == 3:
            W = W[b]

        # GT boxes from item['grid_gt_3d']
        # cols: 0 mem_x 1 mem_y 2 world_x 3 world_y 4 z_base 5 cow_id
        #       6 yaw   7 length 8 width  9 height 10 posture
        n_gt = 0
        g3d = item.get('grid_gt_3d', None)
        if g3d is not None and g3d.shape[-1] >= 11:
            g = g3d[b].detach().float().cpu().numpy()
            g = g[g[:, 5] > 0]  # real entries only
            for row in g:
                mem_xy = torch.tensor(
                    row[0:2], dtype=torch.float32
                ).view(1, 1, 2)
                ref_xy = obb.mem_to_ref_xy(
                    self.vox_util, mem_xy, self.Y, self.Z, self.X
                )[0, 0]
                poly = self._obb_footprint_mem(
                    float(ref_xy[0]), float(ref_xy[1]),
                    float(row[6]), float(row[7]), float(row[8]), W,
                )
                ax_gt.add_patch(_MplPolygon(
                    poly, closed=True, fill=False,
                    edgecolor=self._OBB_GT_COLOR,
                    linewidth=self._OBB_LINEWIDTH,
                    linestyle='-' if row[10] > 0.5 else '--',
                ))
                n_gt += 1

        # Predicted boxes via the shared decode_obb()
        n_pred = 0
        head_keys = ('instance_center', 'instance_offset')
        # NB: the attr_fn closure captures attr_feat, so it must be built
        # from the BATCH-SLICED feature map for this single-image decode.
        attr_fn_b = self._attr_fn_3d(
            {'attr_feat': output['attr_feat'][b:b + 1]}
        ) if output.get('attr_feat') is not None else None
        if all(output.get(k) is not None for k in head_keys) \
                and attr_fn_b is not None:
            out_b = {k: output[k][b:b + 1] for k in head_keys}
            item_b = {
                'frame': item['frame'][b:b + 1].detach().cpu(),
                'worldcoord_from_worldgrid': W,
            }
            if 'sequence_num' in item:
                item_b['sequence_num'] = (
                    item['sequence_num'][b:b + 1].detach().cpu()
                )

            records = obb.decode_obb(
                out_b, item_b, self.vox_util,
                conf_threshold=self.conf_threshold,  # same gate as test_step
                has_3d_gt=True,
                attr_fn=attr_fn_b,
                Y=self.Y, Z=self.Z, X=self.X,
                size_scale_cm=self.size_scale_cm,
                max_detections=self.max_detections,
                coord_space='grid',  # world-grid cells -> Ref2Mem below
                use_global_frame_id=True,
                **self._nms_kwargs(item),
            )
            # records are already score-ordered and conf-filtered; the
            # Top-K cap inside decode_obb keeps the panel readable.
            for r in records[:self.max_detections]:
                _fid, gx, gy, _zb, yaw, l_cm, w_cm, _h, pcls = r[:9]
                poly = self._obb_footprint_mem(gx, gy, yaw, l_cm, w_cm, W)
                ax_pred.add_patch(_MplPolygon(
                    poly, closed=True, fill=False,
                    edgecolor=self._OBB_PRED_COLOR,
                    linewidth=self._OBB_LINEWIDTH,
                    linestyle='-' if int(pcls) == 1 else '--',
                ))
                n_pred += 1

        return n_gt, n_pred

    def plot_data(self, target, output, item=None, batch_idx=0):
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

        # OBB overlay (3D runs only; 2D runs are bit-identical
        #    to the previous behaviour)
        n_gt, n_pred = 0, 0
        if self.has_3d_gt and item is not None:
            try:
                n_gt, n_pred = self._draw_obb_overlay(ax1, ax2, item, output)
            except Exception as e:  # never kill a val epoch
                print(f"[plot] OBB overlay skipped "
                      f"({type(e).__name__}: {e})")
                n_gt, n_pred = 0, 0
            if n_gt or n_pred:
                ax1.set_title(f'center_g + GT OBB (n={n_gt})')
                ax2.set_title(f'center_e + pred OBB (n={n_pred})')
                fig.suptitle(
                    'OBB overlay — GT: lime / pred: red · '
                    'solid = standing, dashed = lying · '
                    f'pred conf > {self.conf_threshold}'
                )
        # exposed for tests / debugging only
        self._last_plot_box_counts = (n_gt, n_pred)

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
        Calibration warmup schedule.
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

    @torch.no_grad()
    def on_before_optimizer_step(self, optimizer):
        """Diagnostic: un-clipped gradient norm per parameter group.

        Lightning calls this BEFORE configure_gradient_clipping, so these
        numbers show exactly what trainer.gradient_clip_val is rescaling.
        If gnorm/head_size or gnorm/calib is >> gnorm/head_center, the
        auxiliary 3D terms are starving the detector.
        """
        if self.global_step % 25 != 0:
            return

        def gnorm(params):
            s = 0.0
            for p in params:
                if p.grad is not None:
                    s += float(p.grad.detach().float().pow(2).sum())
            return s ** 0.5

        dec = self.model.decoder
        total = gnorm(self.parameters())
        self.log('gnorm/total', total)
        # effective clip factor Lightning applies (gradient_clip_val=0.5):
        gcv = float(getattr(self.trainer, 'gradient_clip_val', 0.0) or 0.0)
        self.log('gnorm/clip_scale',
                 min(1.0, gcv / total) if (gcv > 0 and total > 0) else 1.0)
        self.log('gnorm/encoder', gnorm(self.model.encoder.parameters()))
        for name in ('center', 'offset'):
            if name in dec.bev_heads:
                self.log(f'gnorm/head_{name}',
                         gnorm(dec.bev_heads[name].parameters()))
        # yaw/size/posture supervision lives in the query head now —
        # the old loop logged NOTHING for the 3D path.
        self.log('gnorm/attr_neck', gnorm(dec.attr_neck.parameters()))
        self.log('gnorm/attr_head', gnorm(dec.attr3d_head.parameters()))
        for wname in ('center_weight', 'offset_weight', 'tracking_weight',
                      'yaw_weight', 'size_weight', 'posture_weight'):
            p = getattr(self.model, wname, None)
            if p is not None and p.grad is not None:
                self.log(f'gnorm/logvar_{wname}', float(p.grad.abs()))
        if self.learn_calibration:
            self.log('gnorm/calib', gnorm(self.cal_refine.parameters()))

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

        def _parse_ckpt_path(self):
            if not self.config.get("subcommand"):
                return
            cfg = self.config[self.config.subcommand]
            ckpt_path = cfg.get("ckpt_path")
            if not ckpt_path:
                return
            import torch as _torch
            try:
                ckpt = _torch.load(
                    ckpt_path, map_location="cpu", weights_only=True)
            except Exception as e:
                print(f"[CFG-CHECK] WARNING: could not read "
                      f"{ckpt_path} ({e})")
                return
            hp = ckpt.get("hyper_parameters", {})
            if not hp:
                return
            model_cfg = cfg.get("model") or {}
            model_cfg = (model_cfg.as_dict()
                         if hasattr(model_cfg, "as_dict")
                         else dict(model_cfg))

            def _norm(v):
                return list(v) if isinstance(v, (list, tuple)) else v

            differ = [
                (k, hp[k], model_cfg[k]) for k in sorted(hp)
                if not k.startswith("_") and k in model_cfg
                and _norm(model_cfg[k]) != _norm(hp[k])
            ]
            print(f"[CFG-CHECK] ckpt={ckpt_path}")
            print("[CFG-CHECK] checkpoint supplies WEIGHTS ONLY; "
                  "model built from YAML/CLI.")
            if differ:
                print(f"[CFG-CHECK] {len(differ)} model arg(s) DIFFER "
                      f"from the checkpoint's training-time values "
                      f"(intended for test-time ablations; FATAL if "
                      f"accidental):")
                for k, v_train, v_test in differ:
                    print(f"[CFG-CHECK]   {k}: train={v_train!r} -> "
                          f"test={v_test!r}")
            else:
                print("[CFG-CHECK] model config matches the "
                      "checkpoint's training hyperparameters.")

    cli = MyLightningCLI(WorldTrackModel)