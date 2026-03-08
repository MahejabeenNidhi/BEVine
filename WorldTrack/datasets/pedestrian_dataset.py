# WorldTrack/datasets/pedestrian_dataset.py
import os
import json
from operator import itemgetter
import torch
import numpy as np
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F
from PIL import Image
from utils import geom, basic, vox


class PedestrianDataset(VisionDataset):
    def __init__(
        self,
        base,
        is_train=True,
        resolution=(160, 4, 250),
        bounds=(-500, 500, -320, 320, 0, 2),
        final_dim: tuple = (720, 1280),
        resize_lim: list = (0.8, 1.2),
        use_all_frames=False,
        sequence_num=0,
    ):
        super().__init__(base.root)
        self.base = base
        self.root, self.num_cam, self.num_frame = (
            base.root, base.num_cam, base.num_frame
        )
        self.img_shape = base.img_shape
        self.worldgrid_shape = base.worldgrid_shape
        self.is_train = is_train
        self.bounds = bounds
        self.resolution = resolution
        self.use_all_frames = use_all_frames
        self.sequence_num = sequence_num

        self.data_aug_conf = {
            'final_dim': final_dim,
            'resize_lim': resize_lim,
        }
        self.kernel_size = 1.5
        self.max_objects = 60
        self.img_downsample = 4
        self.Y, self.Z, self.X = self.resolution
        self.scene_centroid = torch.tensor(
            (0., 0., 0.)
        ).reshape([1, 3])
        self.vox_util = vox.VoxelUtil(
            self.Y, self.Z, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds,
            assert_cube=False,
        )

        if self.use_all_frames:
            frame_range = range(0, self.num_frame)
        elif self.is_train:
            frame_range = range(0, int(self.num_frame * 0.9))
        else:
            frame_range = range(
                int(self.num_frame * 0.9), self.num_frame
            )

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        self.download(frame_range)

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        self.prepare_gt()
        self.calibration = {}
        self.setup()

    # ------------------------------------------------------------------
    def setup(self):
        intrinsic = torch.tensor(
            np.stack(self.base.intrinsic_matrices, axis=0),
            dtype=torch.float32,
        )
        intrinsic = geom.merge_intrinsics(
            *geom.split_intrinsics(intrinsic)
        ).squeeze()
        self.calibration['intrinsic'] = intrinsic
        self.calibration['extrinsic'] = (
            torch.eye(4)[None].repeat(intrinsic.shape[0], 1, 1)
        )
        self.calibration['extrinsic'][:, :3] = torch.tensor(
            np.stack(self.base.extrinsic_matrices, axis=0),
            dtype=torch.float32,
        )

    def prepare_gt(self):
        og_gt = []
        ann_dir = os.path.join(self.root, 'annotations_positions')
        for fname in sorted(os.listdir(ann_dir)):
            frame = int(fname.split('.')[0])
            with open(os.path.join(ann_dir, fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    v = single_pedestrian['views'][cam]
                    return not (
                        v['xmin'] == -1 and v['xmax'] == -1
                        and v['ymin'] == -1 and v['ymax'] == -1
                    )
                in_cam_range = sum(
                    is_in_cam(cam) for cam in range(self.num_cam)
                )
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(
                    single_pedestrian['positionID']
                )
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0
        ann_dir = os.path.join(self.root, 'annotations_positions')
        for fname in sorted(os.listdir(ann_dir)):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                num_frame += 1
                with open(os.path.join(ann_dir, fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_pts, world_pids = [], []
                img_bboxs = [[] for _ in range(self.num_cam)]
                img_pids = [[] for _ in range(self.num_cam)]
                for pedestrian in all_pedestrians:
                    grid_x, grid_y = (
                        self.base.get_worldgrid_from_pos(
                            pedestrian['positionID']
                        ).squeeze()
                    )
                    if pedestrian['personID'] not in self.pid_dict:
                        self.pid_dict[pedestrian['personID']] = len(
                            self.pid_dict
                        )
                    num_world_bbox += 1
                    world_pts.append((grid_x, grid_y))
                    world_pids.append(pedestrian['personID'])
                    for cam in range(self.num_cam):
                        bbox = itemgetter(
                            'xmin', 'ymin', 'xmax', 'ymax'
                        )(pedestrian['views'][cam])
                        if bbox != (-1, -1, -1, -1):
                            img_bboxs[cam].append(bbox)
                            img_pids[cam].append(
                                pedestrian['personID']
                            )
                            num_imgs_bbox += 1
                self.world_gt[frame] = (
                    torch.tensor(world_pts, dtype=torch.float32),
                    torch.tensor(world_pids, dtype=torch.float32),
                )
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # torch.tensor([]) produces shape (0,) which
                    # breaks 2D indexing in get_img_gt. Explicitly use
                    # shape (0, 4) when no bboxes are visible in this cam.
                    if img_bboxs[cam]:
                        bbox_tensor = torch.tensor(
                            img_bboxs[cam], dtype=torch.float32
                        )  # shape (N, 4)
                    else:
                        bbox_tensor = torch.zeros(
                            (0, 4), dtype=torch.float32
                        )  # shape (0, 4), not (0,)
                    self.imgs_gt[frame][cam] = (
                        bbox_tensor,
                        torch.tensor(img_pids[cam]),
                    )

    # ------------------------------------------------------------------
    # BEV / image GT helpers
    # ------------------------------------------------------------------
    def get_bev_gt(self, mem_pts, mem_pts_prev, pids, pids_pre):
        center = torch.zeros(
            (1, self.Y, self.X), dtype=torch.float32
        )
        valid_mask = torch.zeros(
            (1, self.Y, self.X), dtype=torch.bool
        )
        offset = torch.zeros(
            (4, self.Y, self.X), dtype=torch.float32
        )
        person_ids = torch.zeros(
            (1, self.Y, self.X), dtype=torch.long
        )
        prev_pts = dict(
            zip(pids_pre.int().tolist(), mem_pts_prev[0])
        )
        for pts, pid in zip(mem_pts[0], pids):
            ct = pts[:2]
            ct_int = ct.int()
            if (ct_int[0] < 0 or ct_int[0] >= self.X
                    or ct_int[1] < 0 or ct_int[1] >= self.Y):
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
        H = int(
            self.data_aug_conf['final_dim'][0] / self.img_downsample
        )
        W = int(
            self.data_aug_conf['final_dim'][1] / self.img_downsample
        )
        center = torch.zeros((1, H, W), dtype=torch.float32)
        offset = torch.zeros((2, H, W), dtype=torch.float32)
        size = torch.zeros((2, H, W), dtype=torch.float32)
        valid_mask = torch.zeros((1, H, W), dtype=torch.bool)
        person_ids = torch.zeros((1, H, W), dtype=torch.long)

        if img_pts.shape[0] == 0:
            return center, offset, size, person_ids, valid_mask

        xmin = (img_pts[:, 0] * sx - crop[0]) / self.img_downsample
        ymin = (img_pts[:, 1] * sy - crop[1]) / self.img_downsample
        xmax = (img_pts[:, 2] * sx - crop[0]) / self.img_downsample
        ymax = (img_pts[:, 3] * sy - crop[1]) / self.img_downsample

        foot_pts = np.stack(
            ((xmin + xmax) / 2, ymax), axis=1
        )
        foot_pts = torch.tensor(foot_pts, dtype=torch.float32)

        size_pts = np.stack(
            ((xmax - xmin), (ymax - ymin)), axis=1
        )
        size_pts = torch.tensor(size_pts, dtype=torch.float32)

        for pt_idx, (pid, wh) in enumerate(
                zip(img_pids, size_pts)
        ):
            ct = foot_pts[pt_idx]
            ct_int = ct.int()

            if (ct_int[0] < 0 or ct_int[0] >= W
                    or ct_int[1] < 0 or ct_int[1] >= H):
                continue

            basic.draw_umich_gaussian(
                center[0], ct_int, self.kernel_size
            )
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:, ct_int[1], ct_int[0]] = ct - ct_int
            size[:, ct_int[1], ct_int[0]] = wh
            person_ids[:, ct_int[1], ct_int[0]] = pid

        return center, offset, size, person_ids, valid_mask

    # ------------------------------------------------------------------
    # Augmentation helper
    # ------------------------------------------------------------------
    def sample_augmentation(self):
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(
                *self.data_aug_conf['resize_lim']
            )
            resize_dims = (int(fW * resize), int(fH * resize))
            newW, newH = resize_dims
            crop_h = int((newH - fH) / 2)
            crop_w = int((newW - fW) / 2)
            crop_offset = int(
                self.data_aug_conf['resize_lim'][0]
                * self.data_aug_conf['final_dim'][0]
            )
            crop_w = crop_w + int(
                np.random.uniform(-crop_offset, crop_offset)
            )
            crop_h = crop_h + int(
                np.random.uniform(-crop_offset, crop_offset)
            )
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop

    # ------------------------------------------------------------------
    # Image data loader
    # ------------------------------------------------------------------
    def get_image_data(self, frame, cameras):
        imgs, intrins, extrins = [], [], []
        centers, offsets, sizes, pids, valids = [], [], [], [], []
        for cam in cameras:
            img = Image.open(
                self.img_fpaths[cam][frame]
            ).convert('RGB')
            W, H = img.size
            resize_dims, crop = self.sample_augmentation()
            sx = resize_dims[0] / float(W)
            sy = resize_dims[1] / float(H)
            extrin = self.calibration['extrinsic'][cam]
            intrin = self.calibration['intrinsic'][cam]
            intrin = geom.scale_intrinsics(
                intrin.unsqueeze(0), sx, sy
            ).squeeze(0)
            fx, fy, x0, y0 = geom.split_intrinsics(
                intrin.unsqueeze(0)
            )
            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]
            pix_T_cam = geom.merge_intrinsics(
                fx, fy, new_x0, new_y0
            )
            intrin = pix_T_cam.squeeze(0)
            img = basic.img_transform(img, resize_dims, crop)
            imgs.append(F.to_tensor(img))
            intrins.append(intrin)
            extrins.append(extrin)

            img_pts, img_pids = self.imgs_gt[frame][cam]
            center_img, offset_img, size_img, pid_img, valid_img = (
                self.get_img_gt(img_pts, img_pids, sx, sy, crop)
            )
            centers.append(center_img)
            offsets.append(offset_img)
            sizes.append(size_img)
            pids.append(pid_img)
            valids.append(valid_img)

        return (
            torch.stack(imgs), torch.stack(intrins),
            torch.stack(extrins), torch.stack(centers),
            torch.stack(offsets), torch.stack(sizes),
            torch.stack(pids), torch.stack(valids),
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.world_gt.keys())

    # ------------------------------------------------------------------
    def __getitem__(self, index):
        frame = list(self.world_gt.keys())[index]
        pre_frame = list(self.world_gt.keys())[max(index - 1, 0)]
        cameras = list(range(self.num_cam))

        (imgs, intrins, extrins,
         centers_img, offsets_img, sizes_img,
         pids_img, valids_img) = self.get_image_data(frame, cameras)

        worldcoord_from_worldgrid = torch.eye(4)
        worldcoord_from_worldgrid2d = torch.tensor(
            self.base.worldcoord_from_worldgrid_mat,
            dtype=torch.float32,
        )
        worldcoord_from_worldgrid[:2, :2] = (
            worldcoord_from_worldgrid2d[:2, :2]
        )
        worldcoord_from_worldgrid[:2, 3] = (
            worldcoord_from_worldgrid2d[:2, 2]
        )
        worldgrid_T_worldcoord = torch.inverse(
            worldcoord_from_worldgrid
        )

        worldgrid_pts_org, world_pids = self.world_gt[frame]
        worldgrid_pts_pre, world_pid_pre = self.world_gt[pre_frame]

        worldgrid_pts = torch.cat(
            (worldgrid_pts_org,
             torch.zeros_like(worldgrid_pts_org[:, 0:1])),
            dim=1,
        ).unsqueeze(0)

        worldgrid_pts_pre = torch.cat(
            (worldgrid_pts_pre,
             torch.zeros_like(worldgrid_pts_pre[:, 0:1])),
            dim=1,
        ).unsqueeze(0)

        if self.is_train:
            Rz = torch.eye(3)
            scene_center = torch.tensor(
                [0., 0., 0.], dtype=torch.float32
            )
            off = 0.25
            scene_center[:2].uniform_(-off, off)
            augment = geom.merge_rt(
                Rz.unsqueeze(0), -scene_center.unsqueeze(0)
            ).squeeze()
            worldgrid_T_worldcoord = torch.matmul(
                augment, worldgrid_T_worldcoord
            )
            worldgrid_pts = geom.apply_4x4(
                augment.unsqueeze(0), worldgrid_pts
            )
            worldgrid_pts_pre = geom.apply_4x4(
                augment.unsqueeze(0), worldgrid_pts_pre
            )

        mem_pts = self.vox_util.Ref2Mem(
            worldgrid_pts, self.Y, self.Z, self.X
        )
        mem_pts_pre = self.vox_util.Ref2Mem(
            worldgrid_pts_pre, self.Y, self.Z, self.X
        )

        center_bev, valid_bev, pid_bev, offset_bev = (
            self.get_bev_gt(
                mem_pts, mem_pts_pre, world_pids, world_pid_pre
            )
        )

        grid_gt = torch.zeros(
            (self.max_objects, 3), dtype=torch.long
        )
        grid_gt[:worldgrid_pts.shape[1], :2] = worldgrid_pts_org
        grid_gt[:worldgrid_pts.shape[1], 2] = world_pids

        img_gt_2d = torch.zeros(
            (self.num_cam, self.max_objects, 5), dtype=torch.float32
        )
        for cam in cameras:
            bboxes, pids_cam = self.imgs_gt[frame][cam]
            n = min(len(bboxes), self.max_objects)
            if n > 0:
                img_gt_2d[cam, :n, :4] = bboxes[:n].float()
                img_gt_2d[cam, :n, 4] = pids_cam[:n].float()

        img_paths_str = '||'.join(
            self.img_fpaths[cam][frame] for cam in cameras
        )

        item = {
            'img': imgs,
            'intrinsic': intrins,
            'extrinsic': extrins,
            'ref_T_global': worldgrid_T_worldcoord,
            'frame': frame // self.base.frame_step,
            'sequence_num': int(self.sequence_num),
            'grid_gt': grid_gt,
            'intrinsic_original': self.calibration[
                'intrinsic'
            ].clone(),
            'img_gt_2d': img_gt_2d,
            'img_paths': img_paths_str,
            # BASE (non-augmented) grid→world 3×3 matrix for reprojection loss
            'worldcoord_from_worldgrid': worldcoord_from_worldgrid2d,
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
        }
        return item, target