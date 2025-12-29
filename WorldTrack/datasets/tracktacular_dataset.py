import os
import numpy as np
import cv2
import json
from torchvision.datasets import VisionDataset


class TrackTacular(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        self.__name__ = 'TrackTacular'

        # Image and grid specifications
        self.img_shape = [2160, 3840]  # H, W
        self.worldgrid_shape = [200, 120]  # N_row (Y), N_col (X) - grid cells
        self.num_cam = 8
        self.num_frame = len([f for f in os.listdir(os.path.join(root, 'annotations_positions'))
                              if f.endswith('.json')])
        self.frame_step = 1

        # Grid cell size is 10cm, area is 1200cm x 2000cm
        # Grid coordinates: X goes from 0 to 119 (1200cm), Y goes from 0 to 199 (2000cm)
        # Position ID = grid_x * 200 + grid_y
        # World coordinates in cm
        self.worldcoord_from_worldgrid_mat = np.array([
            [10.0, 0.0, 0.0],  # X: grid to world (10cm per grid cell)
            [0.0, 10.0, 0.0],  # Y: grid to world (10cm per grid cell)
            [0.0, 0.0, 1.0]
        ])

        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_image_fpaths(self, frame_range):
        """Get image file paths for all cameras and frames."""
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        cam_names = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

        for cam_idx, cam_name in enumerate(cam_names):
            cam_folder = os.path.join(self.root, 'Image_subsets', cam_name)
            for fname in sorted(os.listdir(cam_folder)):
                if fname.endswith('.jpg'):
                    frame = int(fname.split('.')[0])
                    if frame in frame_range:
                        img_fpaths[cam_idx][frame] = os.path.join(cam_folder, fname)

        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        """Convert position ID to grid coordinates.

        Position ID = grid_x * 200 + grid_y
        where grid_x is in [0, 119] and grid_y is in [0, 199]
        """
        grid_height = 200  # Y dimension
        grid_x = pos // grid_height
        grid_y = pos % grid_height
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        cam_names = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
        cam_name = cam_names[camera_i]

        # Load intrinsic
        intrinsic_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero', 'intrinsic.txt')
        with open(intrinsic_path, 'r') as f:
            intrinsic_data = eval(f.read())
        intrinsic_matrix = np.array(intrinsic_data['camera_matrix'], dtype=np.float32)

        # Load extrinsic
        extrinsic_path = os.path.join(self.root, 'calibrations', 'extrinsic', f'{cam_name}_extrinsic.txt')
        with open(extrinsic_path, 'r') as f:
            lines = f.readlines()

        rmat_start = lines.index("Rotation Matrix:\n") + 1
        tvec_start = lines.index("Translation Vector (tvec):\n") + 1

        rotation_matrix = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            values = lines[rmat_start + i].strip().split()
            rotation_matrix[i] = [float(v) for v in values]

        tvec = np.zeros(3, dtype=np.float32)
        for i in range(3):
            tvec[i] = float(lines[tvec_start + i].strip())

        # FIXED: Use [R|t] directly (cam_T_world)
        extrinsic_matrix = np.hstack((rotation_matrix, tvec.reshape(3, 1)))

        return intrinsic_matrix, extrinsic_matrix