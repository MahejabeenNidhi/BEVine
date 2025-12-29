import os
import json
from operator import itemgetter
import torch
import numpy as np
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F
from PIL import Image
from utils import geom, basic, vox
from utils.debug_logger import DebugLogger
from collections import defaultdict


class PedestrianDataset(VisionDataset):
    def __init__(
            self,
            base,
            is_train=True,
            resolution=(160, 4, 250),
            bounds=(-500, 500, -320, 320, 0, 2),
            final_dim: tuple = None,
            resize_lim: list = (0.8, 1.2),
            debug=True,
    ):
        super().__init__(base.root)
        self.debug = debug
        if self.debug:
            self.logger = DebugLogger(os.path.join(base.root, 'debug_logs'))
            self.logger.log(f"Initializing PedestrianDataset")
            self.logger.log(f"  is_train: {is_train}")
            self.logger.log(f"  resolution (Y,Z,X): {resolution}")
            self.logger.log(f"  bounds: {bounds}")
            self.logger.log(f"  Dataset: {base.__name__}")
            self.logger.log(f"  Image shape: {base.img_shape}")
            self.logger.log(f"  Grid shape: {base.worldgrid_shape}")
            self.logger.log(f"  Num cameras: {base.num_cam}")
            self.logger.log(f"  Num frames: {base.num_frame}")

        # Compute final_dim from image shape if not provided
        # Compute final_dim based on image aspect ratio
        # Ensure dimensions are divisible by 32 (img_downsample * encoder_downsample)
        if final_dim is None:
            H, W = base.img_shape
            aspect_ratio = W / H

            # Choose height that's divisible by 32
            # For 2800 height -> use 704 (2800 * 0.25 ≈ 700, round to nearest 32)
            target_h = 704
            target_w = int(target_h * aspect_ratio)
            # Round width to nearest 32
            target_w = int(round(target_w / 32) * 32)

            final_dim = (target_h, target_w)

            if self.debug:
                print(f"Auto-computed final_dim: {final_dim} from image shape {base.img_shape}")
                print(f"  Aspect ratio: {aspect_ratio:.3f}")
                print(f"  H divisible by 32: {final_dim[0] % 32 == 0}")
                print(f"  W divisible by 32: {final_dim[1] % 32 == 0}")

        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape = base.img_shape
        self.worldgrid_shape = base.worldgrid_shape
        self.is_train = is_train
        self.bounds = bounds
        self.resolution = resolution
        self.data_aug_conf = {'final_dim': final_dim, 'resize_lim': resize_lim}
        self.kernel_size = 1.5
        self.max_objects = 60
        self.img_downsample = 4
        self.Y, self.Z, self.X = self.resolution
        self.scene_centroid = torch.tensor((0., 0., 0.)).reshape([1, 3])
        self.vox_util = vox.VoxelUtil(
            self.Y, self.Z, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds,
            assert_cube=False)

        # Multi-sequence support
        self.is_multi_sequence = hasattr(self.base, 'sequences')

        # For multi-sequence datasets, sequences are already split into train/val/test
        # For single-sequence datasets, apply the 90/10 split
        if self.is_multi_sequence:
            # Use all frames from the sequences that were loaded
            # The split is already determined by which sequences were loaded in TrackTacularMultiSeq
            frame_range = range(0, self.num_frame)
            if self.debug:
                self.logger.log(f"  Multi-sequence mode: using all {self.num_frame} frames from loaded sequences")
        else:
            # Original single-sequence logic
            if self.is_train:
                frame_range = range(0, int(self.num_frame * 0.9))
            else:
                frame_range = range(int(self.num_frame * 0.9), self.num_frame)
            if self.debug:
                self.logger.log(f"  Single-sequence mode: using frames {min(frame_range)}-{max(frame_range)}")

        self.img_fpaths = self.base.get_image_fpaths(frame_range)

        # Store data per sequence if multi-sequence
        if self.is_multi_sequence:
            self.world_gt = defaultdict(dict)  # {seq_id: {frame: data}}
            self.imgs_gt = defaultdict(dict)  # {seq_id: {frame: data}}
            self.pid_dict = defaultdict(dict)  # {seq_id: {person_id: idx}}
            self.sequence_frames = defaultdict(list)  # {seq_id: [frames]}
        else:
            self.world_gt = {}
            self.imgs_gt = {}
            self.pid_dict = {}

        self.download(frame_range)
        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not self.is_multi_sequence:
            self.prepare_gt()
        else:
            self.prepare_gt_multiseq()

        self.calibration = {}
        self.setup()

    def has_ground_truth(self, seq_id=None):
        """
        Check if ground truth is available for a sequence
        Args:
            seq_id: Sequence ID (for multi-sequence datasets). None for single-sequence.
        Returns:
            bool: True if GT is available
        """
        if not self.is_multi_sequence:
            # Single-sequence datasets always have GT
            return True

        if seq_id is None:
            # Check if any sequence has GT
            return any(self.base.has_ground_truth(s['seq_id']) for s in self.base.sequences)

        # Check specific sequence
        return self.base.has_ground_truth(seq_id)

    def get_image_data_multiseq(self, frame, cameras, imgs_gt_dict):
        """Get image data for multi-sequence dataset"""
        imgs, intrins, extrins = [], [], []
        centers, offsets, sizes, pids, valids = [], [], [], [], []

        for cam in cameras:
            img = Image.open(self.img_fpaths[cam][frame]).convert('RGB')
            W, H = img.size
            resize_dims, crop = self.sample_augmentation()
            sx = resize_dims[0] / float(W)
            sy = resize_dims[1] / float(H)

            extrin = self.calibration['extrinsic'][cam]
            intrin = self.calibration['intrinsic'][cam]
            intrin = geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)
            fx, fy, x0, y0 = geom.split_intrinsics(intrin.unsqueeze(0))
            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]
            pix_T_cam = geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)

            img = basic.img_transform(img, resize_dims, crop)
            imgs.append(F.to_tensor(img))
            intrins.append(intrin)
            extrins.append(extrin)

            img_pts, img_pids = imgs_gt_dict[frame][cam]
            center_img, offset_img, size_img, pid_img, valid_img = \
                self.get_img_gt(img_pts, img_pids, sx, sy, crop)

            centers.append(center_img)
            offsets.append(offset_img)
            sizes.append(size_img)
            pids.append(pid_img)
            valids.append(valid_img)

        return (torch.stack(imgs), torch.stack(intrins), torch.stack(extrins),
                torch.stack(centers), torch.stack(offsets), torch.stack(sizes),
                torch.stack(pids), torch.stack(valids))

    def setup(self):
        """Setup calibration matrices"""
        if self.is_multi_sequence:
            # Use calibration from first sequence as default
            # Actual per-sequence calibration will be loaded in __getitem__
            intrinsic = torch.tensor(
                np.stack(self.base.sequences[0]['intrinsic_matrices'], axis=0),
                dtype=torch.float32
            )
            intrinsic = geom.merge_intrinsics(*geom.split_intrinsics(intrinsic)).squeeze()
            self.calibration['intrinsic'] = intrinsic

            self.calibration['extrinsic'] = torch.eye(4)[None].repeat(intrinsic.shape[0], 1, 1)
            self.calibration['extrinsic'][:, :3] = torch.tensor(
                np.stack(self.base.sequences[0]['extrinsic_matrices'], axis=0),
                dtype=torch.float32
            )
        else:
            intrinsic = torch.tensor(
                np.stack(self.base.intrinsic_matrices, axis=0),
                dtype=torch.float32
            )
            intrinsic = geom.merge_intrinsics(*geom.split_intrinsics(intrinsic)).squeeze()
            self.calibration['intrinsic'] = intrinsic

            self.calibration['extrinsic'] = torch.eye(4)[None].repeat(intrinsic.shape[0], 1, 1)
            self.calibration['extrinsic'][:, :3] = torch.tensor(
                np.stack(self.base.extrinsic_matrices, axis=0),
                dtype=torch.float32
            )

    def prepare_gt(self):
        """Prepare ground truth for single-sequence dataset"""
        ann_path = os.path.join(self.root, 'annotations_positions')

        # Check if annotations exist
        if not os.path.exists(ann_path):
            if self.debug:
                self.logger.log(f"No annotations found at {ann_path} - skipping GT preparation")
            return

        og_gt = []
        for fname in sorted(os.listdir(ann_path)):
            if not fname.endswith('.json'):
                continue

            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)

            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue

                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))

        if len(og_gt) > 0:
            og_gt = np.stack(og_gt, axis=0)
            os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
            np.savetxt(self.gt_fpath, og_gt, '%d')

            if self.debug:
                self.logger.log(f"Saved ground truth file with {len(og_gt)} entries")
        else:
            if self.debug:
                self.logger.log("No ground truth data to save")

    def prepare_gt_multiseq(self):
        """Prepare ground truth for multi-sequence dataset"""
        og_gt = []

        for seq_info in self.base.sequences:
            seq_id = seq_info['seq_id']
            seq_path = seq_info['seq_path']
            has_gt = seq_info.get('has_gt', True)  # NEW: Check if sequence has GT

            # Skip sequences without ground truth
            if not has_gt:
                if self.debug:
                    self.logger.log(f"Skipping GT preparation for sequence {seq_id} (no annotations)")
                continue

            ann_path = os.path.join(seq_path, 'annotations_positions')

            # Double-check that annotation path exists
            if not os.path.exists(ann_path):
                if self.debug:
                    self.logger.log(f"Warning: Annotation path does not exist for sequence {seq_id}: {ann_path}")
                continue

            for fname in sorted(os.listdir(ann_path)):
                if not fname.endswith('.json'):
                    continue

                local_frame = int(fname.split('.')[0])
                global_frame = seq_info['frame_offset'] + local_frame - 1

                # Check if this frame was loaded in download()
                if seq_id not in self.world_gt or global_frame not in self.world_gt[seq_id]:
                    continue

                with open(os.path.join(ann_path, fname)) as json_file:
                    all_pedestrians = json.load(json_file)

                for pedestrian in all_pedestrians:
                    def is_in_cam(cam):
                        return not (pedestrian['views'][cam]['xmin'] == -1 and
                                    pedestrian['views'][cam]['xmax'] == -1 and
                                    pedestrian['views'][cam]['ymin'] == -1 and
                                    pedestrian['views'][cam]['ymax'] == -1)

                    in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                    if not in_cam_range:
                        continue

                    grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID'])
                    og_gt.append(np.array([seq_id, global_frame, grid_x, grid_y]))

        # Only save GT file if we have any ground truth data
        if len(og_gt) > 0:
            og_gt = np.stack(og_gt, axis=0)
            os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
            np.savetxt(self.gt_fpath, og_gt, '%d')

            if self.debug:
                self.logger.log(f"Saved ground truth file with {len(og_gt)} entries")
        else:
            if self.debug:
                self.logger.log("No ground truth data to save")

    def download(self, frame_range):
        """Download annotations for specified frame range"""
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0

        if self.debug:
            self.logger.log("\n=== Starting annotation download ===")
            self.logger.log(f"Frame range: {min(frame_range)} to {max(frame_range)}")
            self.logger.log(f"Multi-sequence: {self.is_multi_sequence}")

        if self.is_multi_sequence:
            # Process each sequence separately
            for seq_info in self.base.sequences:
                seq_id = seq_info['seq_id']
                seq_path = seq_info['seq_path']
                has_gt = seq_info['has_gt']  # NEW: Check if GT exists

                if self.debug:
                    gt_status = "WITH GT" if has_gt else "WITHOUT GT"
                    self.logger.log(f"\nProcessing sequence {seq_id}: {seq_info['seq_name']} [{gt_status}]")

                if not has_gt:
                    # ====== NO GROUND TRUTH - Create empty placeholders ======
                    if self.debug:
                        self.logger.log(f"  No annotations found - creating empty GT placeholders")

                    # Process all frames with empty GT
                    for local_frame_idx in range(seq_info['num_frames']):
                        global_frame_idx = seq_info['frame_offset'] + local_frame_idx
                        num_frame += 1
                        self.sequence_frames[seq_id].append(global_frame_idx)

                        # Empty world GT
                        self.world_gt[seq_id][global_frame_idx] = (
                            torch.zeros((0, 2), dtype=torch.float32),  # No world points
                            torch.zeros((0,), dtype=torch.float32)  # No person IDs
                        )

                        # Empty image GT for all cameras
                        self.imgs_gt[seq_id][global_frame_idx] = {}
                        for cam in range(self.num_cam):
                            self.imgs_gt[seq_id][global_frame_idx][cam] = (
                                torch.zeros((0, 4), dtype=torch.float32),  # No bboxes
                                torch.zeros((0,), dtype=torch.float32)  # No person IDs
                            )

                    if self.debug:
                        self.logger.log(f"  Created empty GT for {seq_info['num_frames']} frames")

                    continue  # Skip to next sequence

                # ====== HAS GROUND TRUTH - Process normally ======
                ann_path = os.path.join(seq_path, 'annotations_positions')

                for fname in sorted(os.listdir(ann_path)):
                    if not fname.endswith('.json'):
                        continue

                    local_frame = int(fname.split('.')[0])
                    global_frame = seq_info['frame_offset'] + local_frame - 1
                    num_frame += 1
                    self.sequence_frames[seq_id].append(global_frame)

                    with open(os.path.join(ann_path, fname)) as json_file:
                        all_pedestrians = json.load(json_file)

                    world_pts, world_pids = [], []
                    img_bboxs = [[] for _ in range(self.num_cam)]
                    img_pids = [[] for _ in range(self.num_cam)]

                    for pedestrian in all_pedestrians:
                        grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                        person_id = pedestrian['personID']

                        if person_id not in self.pid_dict[seq_id]:
                            self.pid_dict[seq_id][person_id] = len(self.pid_dict[seq_id])

                        num_world_bbox += 1
                        world_pts.append((grid_x, grid_y))
                        world_pids.append(person_id)

                        for cam in range(self.num_cam):
                            bbox = pedestrian['views'][cam]
                            if (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']) != (-1, -1, -1, -1):
                                img_bboxs[cam].append((bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']))
                                img_pids[cam].append(person_id)
                                num_imgs_bbox += 1

                    self.world_gt[seq_id][global_frame] = (
                        torch.tensor(world_pts, dtype=torch.float32),
                        torch.tensor(world_pids, dtype=torch.float32)
                    )

                    self.imgs_gt[seq_id][global_frame] = {}
                    for cam in range(self.num_cam):
                        if len(img_bboxs[cam]) > 0:
                            self.imgs_gt[seq_id][global_frame][cam] = (
                                torch.tensor(img_bboxs[cam], dtype=torch.float32),
                                torch.tensor(img_pids[cam], dtype=torch.float32)
                            )
                        else:
                            self.imgs_gt[seq_id][global_frame][cam] = (
                                torch.zeros((0, 4), dtype=torch.float32),
                                torch.zeros((0,), dtype=torch.float32)
                            )

                if self.debug:
                    self.logger.log(f"  Sequence {seq_id} frames: {len(self.sequence_frames[seq_id])}")

        else:
            # Original single-sequence logic (unchanged)
            ann_path = os.path.join(self.root, 'annotations_positions')
            for fname in sorted(os.listdir(ann_path)):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    num_frame += 1
                    with open(os.path.join(ann_path, fname)) as json_file:
                        all_pedestrians = json.load(json_file)

                    world_pts, world_pids = [], []
                    img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]

                    for pedestrian in all_pedestrians:
                        grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()

                        if pedestrian['personID'] not in self.pid_dict:
                            self.pid_dict[pedestrian['personID']] = len(self.pid_dict)

                        num_world_bbox += 1
                        world_pts.append((grid_x, grid_y))
                        world_pids.append(pedestrian['personID'])

                        for cam in range(self.num_cam):
                            bbox = pedestrian['views'][cam]
                            if (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']) != (-1, -1, -1, -1):
                                img_bboxs[cam].append((bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']))
                                img_pids[cam].append(pedestrian['personID'])
                                num_imgs_bbox += 1

                    self.world_gt[frame] = (torch.tensor(world_pts, dtype=torch.float32),
                                            torch.tensor(world_pids, dtype=torch.float32))
                    self.imgs_gt[frame] = {}
                    for cam in range(self.num_cam):
                        if len(img_bboxs[cam]) > 0:
                            self.imgs_gt[frame][cam] = (
                                torch.tensor(img_bboxs[cam], dtype=torch.float32),
                                torch.tensor(img_pids[cam], dtype=torch.float32)
                            )
                        else:
                            self.imgs_gt[frame][cam] = (
                                torch.zeros((0, 4), dtype=torch.float32),
                                torch.zeros((0,), dtype=torch.float32)
                            )

        if self.debug:
            self.logger.log(f"\nDownload complete:")
            self.logger.log(f"  Frames processed: {num_frame}")
            self.logger.log(f"  World bboxes: {num_world_bbox}")
            self.logger.log(f"  Image bboxes: {num_imgs_bbox}")

    def get_bev_gt(self, mem_pts, mem_pts_prev, pids, pids_pre):
        center = torch.zeros((1, self.Y, self.X), dtype=torch.float32)
        valid_mask = torch.zeros((1, self.Y, self.X), dtype=torch.bool)
        offset = torch.zeros((4, self.Y, self.X), dtype=torch.float32)
        person_ids = torch.zeros((1, self.Y, self.X), dtype=torch.long)

        prev_pts = dict(zip(pids_pre.int().tolist(), mem_pts_prev[0]))

        for pts, pid in zip(mem_pts[0], pids):
            ct = pts[:2]
            ct_int = ct.int()
            if ct_int[0] < 0 or ct_int[0] >= self.X or ct_int[1] < 0 or ct_int[1] >= self.Y:
                continue

            for c in center:
                basic.draw_umich_gaussian(c, ct_int, self.kernel_size)

            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:2, ct_int[1], ct_int[0]] = ct - ct_int
            person_ids[:, ct_int[1], ct_int[0]] = pid

            if pid in pids_pre:
                t_off = prev_pts[pid.int().item()][:2] - ct_int
                if t_off.abs().max() > 15:
                    continue
                offset[2:, ct_int[1], ct_int[0]] = t_off

        return center, valid_mask, person_ids, offset

    def get_img_gt(self, img_pts, img_pids, sx, sy, crop):
        H = int(self.data_aug_conf['final_dim'][0] / self.img_downsample)
        W = int(self.data_aug_conf['final_dim'][1] / self.img_downsample)

        center = torch.zeros((3, H, W), dtype=torch.float32)
        offset = torch.zeros((2, H, W), dtype=torch.float32)
        size = torch.zeros((2, H, W), dtype=torch.float32)
        valid_mask = torch.zeros((1, H, W), dtype=torch.bool)
        person_ids = torch.zeros((1, H, W), dtype=torch.long)

        xmin = (img_pts[:, 0] * sx - crop[0]) / self.img_downsample
        ymin = (img_pts[:, 1] * sy - crop[1]) / self.img_downsample
        xmax = (img_pts[:, 2] * sx - crop[0]) / self.img_downsample
        ymax = (img_pts[:, 3] * sy - crop[1]) / self.img_downsample

        center_pts = np.stack(((xmin + xmax) / 2, (ymin + ymax) / 2), axis=1)
        center_pts = torch.tensor(center_pts, dtype=torch.float32)

        size_pts = np.stack(((-xmin + xmax), (-ymin + ymax)), axis=1)
        size_pts = torch.tensor(size_pts, dtype=torch.float32)

        foot_pts = np.stack(((xmin + xmax) / 2, ymin), axis=1)
        foot_pts = torch.tensor(foot_pts, dtype=torch.float32)

        head_pts = np.stack(((xmin + xmax) / 2, ymax), axis=1)
        head_pts = torch.tensor(head_pts, dtype=torch.float32)

        for pt_idx, (pid, wh) in enumerate(zip(img_pids, size_pts)):
            for idx, pt in enumerate((foot_pts[pt_idx],)):
                if pt[0] < 0 or pt[0] >= W or pt[1] < 0 or pt[1] >= H:
                    continue
                basic.draw_umich_gaussian(center[idx], pt.int(), self.kernel_size)

            ct_int = foot_pts[pt_idx].int()
            if ct_int[0] < 0 or ct_int[0] >= W or ct_int[1] < 0 or ct_int[1] >= H:
                continue

            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:, ct_int[1], ct_int[0]] = foot_pts[pt_idx] - ct_int
            size[:, ct_int[1], ct_int[0]] = wh
            person_ids[:, ct_int[1], ct_int[0]] = pid

        return center, offset, size, person_ids, valid_mask

    def sample_augmentation(self):
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(fW * resize), int(fH * resize))
            newW, newH = resize_dims
            crop_h = int((newH - fH) / 2)
            crop_w = int((newW - fW) / 2)
            crop_offset = int(self.data_aug_conf['resize_lim'][0] * self.data_aug_conf['final_dim'][0])
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop

    def get_image_data(self, frame, cameras):
        imgs, intrins, extrins = [], [], []
        centers, offsets, sizes, pids, valids = [], [], [], [], []

        for cam in cameras:
            img = Image.open(self.img_fpaths[cam][frame]).convert('RGB')
            W, H = img.size
            resize_dims, crop = self.sample_augmentation()
            sx = resize_dims[0] / float(W)
            sy = resize_dims[1] / float(H)

            extrin = self.calibration['extrinsic'][cam]
            intrin = self.calibration['intrinsic'][cam]
            intrin = geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)
            fx, fy, x0, y0 = geom.split_intrinsics(intrin.unsqueeze(0))
            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]
            pix_T_cam = geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)

            img = basic.img_transform(img, resize_dims, crop)
            imgs.append(F.to_tensor(img))
            intrins.append(intrin)
            extrins.append(extrin)

            img_pts, img_pids = self.imgs_gt[frame][cam]
            center_img, offset_img, size_img, pid_img, valid_img = self.get_img_gt(img_pts, img_pids, sx, sy, crop)

            centers.append(center_img)
            offsets.append(offset_img)
            sizes.append(size_img)
            pids.append(pid_img)
            valids.append(valid_img)

        return torch.stack(imgs), torch.stack(intrins), torch.stack(extrins), torch.stack(centers), torch.stack(
            offsets), torch.stack(sizes), torch.stack(pids), torch.stack(valids)

    def __len__(self):
        if self.is_multi_sequence:
            return sum(len(frames) for frames in self.sequence_frames.values())
        else:
            return len(self.world_gt.keys())

    def __getitem__(self, index):
        # Get the frame and sequence ID
        if self.is_multi_sequence:
            # Find which sequence this index belongs to
            cumulative = 0
            for seq_id in sorted(self.sequence_frames.keys()):
                seq_frames = sorted(self.sequence_frames[seq_id])
                if index < cumulative + len(seq_frames):
                    local_idx = index - cumulative
                    frame = seq_frames[local_idx]
                    pre_frame = seq_frames[max(local_idx - 1, 0)]

                    if not hasattr(self.base, 'seq_id_to_index'):
                        raise AttributeError(
                            f"Base dataset {type(self.base).__name__} missing seq_id_to_index mapping. "
                            f"Please add it in the dataset class."
                        )

                    seq_list_index = self.base.seq_id_to_index[seq_id]
                    seq_info = self.base.sequences[seq_list_index]

                    # Get sequence-specific calibration
                    intrinsic = torch.tensor(
                        np.stack(seq_info['intrinsic_matrices'], axis=0),
                        dtype=torch.float32
                    )
                    intrinsic = geom.merge_intrinsics(*geom.split_intrinsics(intrinsic)).squeeze()

                    extrinsic = torch.eye(4)[None].repeat(intrinsic.shape[0], 1, 1)
                    extrinsic[:, :3] = torch.tensor(
                        np.stack(seq_info['extrinsic_matrices'], axis=0),
                        dtype=torch.float32
                    )

                    # Update calibration for this sequence
                    self.calibration['intrinsic'] = intrinsic
                    self.calibration['extrinsic'] = extrinsic

                    world_gt_dict = self.world_gt[seq_id]
                    imgs_gt_dict = self.imgs_gt[seq_id]
                    break
                cumulative += len(seq_frames)
        else:
            seq_id = 0
            frame = list(self.world_gt.keys())[index]
            pre_frame = list(self.world_gt.keys())[max(index - 1, 0)]
            world_gt_dict = self.world_gt
            imgs_gt_dict = self.imgs_gt

        cameras = list(range(self.num_cam))

        # Get images
        imgs, intrins, extrins, centers_img, offsets_img, sizes_img, pids_img, valids_img = \
            self.get_image_data_multiseq(frame, cameras, imgs_gt_dict) if self.is_multi_sequence else \
                self.get_image_data(frame, cameras)

        # Compute grid_T_world transformation
        worldcoord_from_worldgrid = torch.eye(4)
        worldcoord_from_worldgrid2d = torch.tensor(
            self.base.worldcoord_from_worldgrid_mat, dtype=torch.float32
        )
        worldcoord_from_worldgrid[:2, :2] = worldcoord_from_worldgrid2d[:2, :2]
        worldcoord_from_worldgrid[:2, 3] = worldcoord_from_worldgrid2d[:2, 2]
        worldgrid_T_worldcoord = torch.inverse(worldcoord_from_worldgrid)

        # Ground truth in grid coordinates
        worldgrid_pts_org, world_pids = world_gt_dict[frame]
        worldgrid_pts_pre, world_pid_pre = world_gt_dict[pre_frame]

        worldgrid_pts = torch.cat(
            (worldgrid_pts_org, torch.zeros_like(worldgrid_pts_org[:, 0:1])),
            dim=1
        ).unsqueeze(0)
        worldgrid_pts_pre = torch.cat(
            (worldgrid_pts_pre, torch.zeros_like(worldgrid_pts_pre[:, 0:1])),
            dim=1
        )

        if self.is_train:
            Rz = torch.eye(3)
            scene_center = torch.tensor([0., 0., 0.], dtype=torch.float32)
            off = 0.25
            scene_center[:2].uniform_(-off, off)
            augment = geom.merge_rt(Rz.unsqueeze(0), -scene_center.unsqueeze(0)).squeeze()
            worldgrid_T_worldcoord = torch.matmul(augment, worldgrid_T_worldcoord)
            worldgrid_pts = geom.apply_4x4(augment.unsqueeze(0), worldgrid_pts)

        # Convert grid coords to memory coords
        mem_pts = self.vox_util.Ref2Mem(worldgrid_pts, self.Y, self.Z, self.X)
        mem_pts_pre = self.vox_util.Ref2Mem(
            worldgrid_pts_pre.unsqueeze(0), self.Y, self.Z, self.X
        )

        center_bev, valid_bev, pid_bev, offset_bev = self.get_bev_gt(
            mem_pts, mem_pts_pre, world_pids, world_pid_pre
        )

        grid_gt = torch.zeros((self.max_objects, 3), dtype=torch.long)
        grid_gt[:worldgrid_pts.shape[1], :2] = worldgrid_pts_org
        grid_gt[:worldgrid_pts.shape[1], 2] = world_pids

        item = {
            'img': imgs,
            'intrinsic': intrins,
            'extrinsic': extrins,
            'ref_T_global': worldgrid_T_worldcoord,
            'frame': frame // self.base.frame_step,
            'sequence_num': int(seq_id),
            'grid_gt': grid_gt,
            'has_gt': self.has_ground_truth(seq_id),
            'dataset_name': getattr(self.base, '__name__', 'unknown'),
            'dataset_id': 0,
            'resolution': tuple(self.resolution),
            'bounds': tuple(self.bounds),
        }

        target = {
            'valid_bev': valid_bev,
            'center_bev': center_bev,
            'offset_bev': offset_bev,
            'pid_bev': pid_bev,
            'center_img': centers_img,
            'offset_img': offsets_img,
            'size_img': sizes_img,
            'valid_img': valids_img,
            'pid_img': pids_img,
            'resolution': tuple(self.resolution),
            'bounds': tuple(self.bounds),
        }

        return item, target