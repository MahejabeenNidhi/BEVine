import os
import numpy as np
import json
from torchvision.datasets import VisionDataset


class TrackTacularMMCows(VisionDataset):
    """Multi-sequence dataset for MMCows with 4 cameras"""

    def __init__(self, root, split='train', sequences_file='sequences.json', sequence_id_offset=1000):
        """
        Args:
            root: Path to TrackTacular_mmCows_MultiSeq directory
            split: 'train', 'val', or 'test'
            sequences_file: Name of the sequences JSON file
        """
        super().__init__(root)
        self.__name__ = 'TrackTacularMMCows'
        self.sequence_id_offset = sequence_id_offset

        # Load sequence metadata
        sequences_path = os.path.join(root, sequences_file)
        if not os.path.exists(sequences_path):
            raise FileNotFoundError(
                f"{sequences_file} not found at {sequences_path}. "
                "Please create this file to specify train/val/test sequences."
            )

        with open(sequences_path, 'r') as f:
            sequences_data = json.load(f)

        self.split = split
        self.sequence_names = sequences_data[split]
        self.sequence_paths = [os.path.join(root, seq_name) for seq_name in self.sequence_names]

        # Verify all sequence directories exist
        for seq_path in self.sequence_paths:
            if not os.path.exists(seq_path):
                raise FileNotFoundError(f"Sequence directory not found: {seq_path}")

        # Image and grid specifications
        self.img_shape = [2800, 4480]  # H, W
        self.num_cam = 4
        self.frame_step = 1

        # Farm bounds in centimeters
        self.x_min_cm = -879
        self.x_max_cm = 1042
        self.y_min_cm = -646
        self.y_max_cm = 533

        # Grid specifications - 10cm per grid cell
        self.grid_cell_size = 10  # cm

        # Calculate grid dimensions
        self.worldgrid_shape = [
            int((self.y_max_cm - self.y_min_cm) / self.grid_cell_size),  # N_row (Y)
            int((self.x_max_cm - self.x_min_cm) / self.grid_cell_size)  # N_col (X)
        ]

        print(f"Grid shape: {self.worldgrid_shape} (Y x X)")
        print(f"X range: {self.x_min_cm} to {self.x_max_cm} cm")
        print(f"Y range: {self.y_min_cm} to {self.y_max_cm} cm")

        # World coordinate transformation matrix
        self.worldcoord_from_worldgrid_mat = np.array([
            [self.grid_cell_size, 0.0, self.x_min_cm],
            [0.0, self.grid_cell_size, self.y_min_cm],
            [0.0, 0.0, 1.0]
        ])

        # Build sequence information
        self.sequences = []
        self.total_frames = 0

        for seq_idx, (seq_name, seq_path) in enumerate(zip(self.sequence_names, self.sequence_paths)):
            # Check if annotations exist for this sequence
            ann_path = os.path.join(seq_path, 'annotations_positions')
            has_gt = os.path.exists(ann_path) and len([f for f in os.listdir(ann_path) if f.endswith('.json')]) > 0

            # Count frames
            if has_gt:
                # Count from annotations
                num_frames = len([f for f in os.listdir(ann_path) if f.endswith('.json')])
            else:
                # Count from images (use first camera)
                img_path = os.path.join(seq_path, 'Image_subsets', 'C1')
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image directory not found: {img_path}")
                num_frames = len([f for f in os.listdir(img_path) if f.endswith('.jpg')])

            # Get calibration for this sequence
            intrinsic_matrices, extrinsic_matrices = zip(
                *[self.get_intrinsic_extrinsic_matrix(seq_path, cam) for cam in range(self.num_cam)]
            )

            self.sequences.append({
                'seq_id': seq_idx + self.sequence_id_offset,
                'seq_name': seq_name,
                'seq_path': seq_path,
                'num_frames': num_frames,
                'frame_offset': self.total_frames,
                'intrinsic_matrices': intrinsic_matrices,
                'extrinsic_matrices': extrinsic_matrices,
                'has_gt': has_gt,  # NEW: Track whether this sequence has GT
            })

            self.total_frames += num_frames

        self.num_frame = self.total_frames

        # Create mapping from seq_id to list index
        self.seq_id_to_index = {seq['seq_id']: i for i, seq in enumerate(self.sequences)}

        # Use first sequence's calibration as default
        self.intrinsic_matrices = self.sequences[0]['intrinsic_matrices']
        self.extrinsic_matrices = self.sequences[0]['extrinsic_matrices']

        # Print summary
        print(f"\n=== TrackTacularMMCows initialized ===")
        print(f"Split: {split}")
        print(f"Sequences file: {sequences_file}")
        print(f"Sequences loaded: {self.sequence_names}")
        print(f"Total frames: {self.total_frames}")
        print(f"Cameras: {self.num_cam}")
        print(f"Image size: {self.img_shape[1]} x {self.img_shape[0]} (W x H)")

        sequences_with_gt = sum(1 for s in self.sequences if s['has_gt'])
        sequences_without_gt = len(self.sequences) - sequences_with_gt

        print(f"\nGround Truth Status:")
        print(f"  Sequences WITH GT: {sequences_with_gt}")
        print(f"  Sequences WITHOUT GT: {sequences_without_gt}")

        for seq_info in self.sequences:
            gt_status = "✓ HAS GT" if seq_info['has_gt'] else "✗ NO GT"
            print(f"  Seq {seq_info['seq_id']} ({seq_info['seq_name']}): {seq_info['num_frames']} frames [{gt_status}]")
        print("=" * 40)

    def get_sequence_info(self, global_frame_idx):
        """Get sequence information for a global frame index"""
        for seq_info in self.sequences:
            if global_frame_idx < seq_info['frame_offset'] + seq_info['num_frames']:
                local_frame_idx = global_frame_idx - seq_info['frame_offset']
                return seq_info, local_frame_idx
        raise IndexError(f"Global frame index {global_frame_idx} out of range")

    def has_ground_truth(self, seq_id):
        """Check if a sequence has ground truth annotations"""
        seq_index = self.seq_id_to_index[seq_id]
        return self.sequences[seq_index]['has_gt']

    def get_image_fpaths(self, frame_range):
        """Get image file paths for all cameras and frames across all sequences"""
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        cam_names = ['C1', 'C2', 'C3', 'C4']

        for seq_info in self.sequences:
            seq_path = seq_info['seq_path']

            for local_frame_idx in range(seq_info['num_frames']):
                global_frame_idx = seq_info['frame_offset'] + local_frame_idx
                frame_num = local_frame_idx + 1  # 1-indexed

                for cam_idx, cam_name in enumerate(cam_names):
                    cam_folder = os.path.join(seq_path, 'Image_subsets', cam_name)
                    fname = f"{frame_num:08d}.jpg"
                    fpath = os.path.join(cam_folder, fname)

                    if os.path.exists(fpath):
                        img_fpaths[cam_idx][global_frame_idx] = fpath

        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        """
        Convert position ID to grid coordinates
        Position ID is calculated as: grid_x * grid_height + grid_y
        """
        grid_height = self.worldgrid_shape[0]  # Number of rows (Y)
        grid_x = pos // grid_height
        grid_y = pos % grid_height
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, seq_path, camera_i):
        """
        Load calibration for a specific camera in a sequence
        Each camera has its own intrinsic matrix
        """
        cam_names = ['C1', 'C2', 'C3', 'C4']
        cam_name = cam_names[camera_i]

        # Load intrinsic - each camera has its own file
        intrinsic_path = os.path.join(
            seq_path, 'calibrations', 'intrinsic_zero', f'{cam_name}_intrinsic.txt'
        )

        if not os.path.exists(intrinsic_path):
            raise FileNotFoundError(f"Intrinsic file not found: {intrinsic_path}")

        with open(intrinsic_path, 'r') as f:
            intrinsic_data = eval(f.read())

        intrinsic_matrix = np.array(intrinsic_data['camera_matrix'], dtype=np.float32)

        # Load extrinsic
        extrinsic_path = os.path.join(
            seq_path, 'calibrations', 'extrinsic', f'{cam_name}_extrinsic.txt'
        )

        if not os.path.exists(extrinsic_path):
            raise FileNotFoundError(f"Extrinsic file not found: {extrinsic_path}")

        with open(extrinsic_path, 'r') as f:
            lines = f.readlines()

        # Parse rotation matrix and translation vector
        rmat_start = None
        tvec_start = None

        for i, line in enumerate(lines):
            if "Rotation Matrix:" in line:
                rmat_start = i + 1
            if "Translation Vector (tvec):" in line:
                tvec_start = i + 1

        if rmat_start is None or tvec_start is None:
            raise ValueError(f"Could not parse extrinsic file: {extrinsic_path}")

        # Read rotation matrix (3x3)
        rotation_matrix = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            values = lines[rmat_start + i].strip().split()
            rotation_matrix[i] = [float(v) for v in values]

        # Read translation vector (3x1)
        tvec = np.zeros(3, dtype=np.float32)
        for i in range(3):
            tvec[i] = float(lines[tvec_start + i].strip())

        # Combine into extrinsic matrix [R|t] (3x4)
        extrinsic_matrix = np.hstack((rotation_matrix, tvec.reshape(3, 1)))

        return intrinsic_matrix, extrinsic_matrix