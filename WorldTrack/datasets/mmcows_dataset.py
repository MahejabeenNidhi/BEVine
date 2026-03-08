# WorldTrack/datasets/mmcows_dataset.py

import os
import ast
import numpy as np
import cv2
from torchvision.datasets import VisionDataset


class MmCows(VisionDataset):
    def __init__(self, root):
        super().__init__(root)

        self.__name__ = 'MmCows'
        self.img_shape = [2800, 4480]  # H, W

        self.x_min_cm = -879
        self.x_max_cm = 1042
        self.y_min_cm = -646
        self.y_max_cm = 533
        self.grid_cell_size = 10

        self.grid_width = int((self.x_max_cm - self.x_min_cm) / self.grid_cell_size)
        self.grid_height = int((self.y_max_cm - self.y_min_cm) / self.grid_cell_size)
        self.worldgrid_shape = [self.grid_height, self.grid_width]

        self.num_cam = 4
        self.num_frame = self._count_frames()
        self.frame_step = 1

        self.worldcoord_from_worldgrid_mat = np.array([
            [self.grid_cell_size, 0, self.x_min_cm],
            [0, self.grid_cell_size, self.y_min_cm],
            [0, 0, 1]
        ], dtype=np.float64)

        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)]
        )

    def _count_frames(self):
        ann_dir = os.path.join(self.root, 'annotations_positions')
        if not os.path.exists(ann_dir):
            raise FileNotFoundError(f"Annotations directory not found: {ann_dir}")
        frame_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
        if not frame_files:
            raise ValueError(f"No annotation JSON files found in {ann_dir}")
        return len(frame_files)

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        cam_names = ['C1', 'C2', 'C3', 'C4']

        img_subsets_dir = os.path.join(self.root, 'Image_subsets')
        for cam_idx, cam_name in enumerate(cam_names):
            cam_dir = os.path.join(img_subsets_dir, cam_name)
            if not os.path.exists(cam_dir):
                continue
            for fname in sorted(os.listdir(cam_dir)):
                if not (fname.endswith('.jpg') or fname.endswith('.png')):
                    continue
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam_idx][frame] = os.path.join(cam_dir, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_y = pos % self.grid_height
        grid_x = pos // self.grid_height
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        cam_names = ['C1', 'C2', 'C3', 'C4']
        cam_name = cam_names[camera_i]

        # ── Intrinsic ──
        intrinsic_path = os.path.join(
            self.root, 'calibrations', 'intrinsic_zero', f'{cam_name}_intrinsic.txt'
        )
        with open(intrinsic_path, 'r') as f:
            intrinsic_data = ast.literal_eval(f.read())
        intrinsic_matrix = np.array(intrinsic_data['camera_matrix'], dtype=np.float32)

        # ── Extrinsic: Try recovered first, fall back to original ──
        recovered_path = os.path.join(
            self.root, 'calibrations', 'extrinsic_recovered',
            f'{cam_name}_extrinsic.npz'
        )

        if os.path.exists(recovered_path):
            data = np.load(recovered_path, allow_pickle=True)
            extrinsic_matrix = data['Rt'].astype(np.float32)
            method = str(data['method'])
            med_err = float(data['median_error'])
            cam_center = data['cam_center']
            print(f"  {cam_name}: Loaded RECOVERED extrinsic "
                  f"(method={method}, median_err={med_err:.1f}px, "
                  f"center=({cam_center[0]:.0f},{cam_center[1]:.0f},{cam_center[2]:.0f}))")
            return intrinsic_matrix, extrinsic_matrix

        # ── Fallback: original file with Rodrigues ──
        print(f"  {cam_name}: ⚠ No recovered extrinsic found — using original (LIKELY BROKEN)")
        print(f"         Run: python recover_extrinsics.py --data_dir {self.root}")

        extrinsic_path = os.path.join(
            self.root, 'calibrations', 'extrinsic', f'{cam_name}_extrinsic.txt'
        )
        with open(extrinsic_path, 'r') as f:
            lines = f.readlines()

        rvec_values = []
        tvec_values = []
        reading = None

        for line in lines:
            stripped = line.strip()
            if 'Rotation Vector' in stripped:
                reading = 'rvec'; continue
            elif 'Translation Vector' in stripped:
                reading = 'tvec'; continue
            elif 'Rotation Matrix' in stripped:
                reading = None; continue
            if not stripped:
                continue
            try:
                if reading == 'rvec' and len(rvec_values) < 3:
                    rvec_values.append(float(stripped))
                elif reading == 'tvec' and len(tvec_values) < 3:
                    tvec_values.append(float(stripped))
            except ValueError:
                continue

        rvec = np.array(rvec_values, dtype=np.float64)
        tvec = np.array(tvec_values, dtype=np.float32)
        R, _ = cv2.Rodrigues(rvec)

        extrinsic_matrix = np.hstack((R.astype(np.float32), tvec.reshape(3, 1)))
        return intrinsic_matrix, extrinsic_matrix