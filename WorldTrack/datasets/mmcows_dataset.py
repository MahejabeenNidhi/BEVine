# WorldTrack/datasets/mmcows_dataset.py

import os
import ast
import json
import numpy as np
import cv2
from torchvision.datasets import VisionDataset


class MmCows(VisionDataset):
    ANN_GRID_CELL_SIZE = 10.0  # cm
    ANN_X_MIN_CM = -879.0
    ANN_Y_MIN_CM = -646.0
    ANN_POS_CELL_CENTER = True  # decode to the CENTRE of the annotation cell
    ANN_STRIDE_AXIS = 'height'  # 'height': pos = gx*H + gy ; 'width': pos = gy*W + gx

    def __init__(self, root):
        super().__init__(root)
        self.__name__ = 'MmCows'
        self.img_shape = [2800, 4480]  # H, W
        self.x_min_cm = -879
        self.x_max_cm = 1042
        self.y_min_cm = -646
        self.y_max_cm = 533

        # runtime BEV grid
        self.grid_cell_size = 10
        self.grid_width = int((self.x_max_cm - self.x_min_cm) / self.grid_cell_size)
        self.grid_height = int((self.y_max_cm - self.y_min_cm) / self.grid_cell_size)
        self.worldgrid_shape = [self.grid_height, self.grid_width]

        # annotation grid
        self.ann_grid_cell_size = float(self.ANN_GRID_CELL_SIZE)
        self.ann_grid_width = int((self.x_max_cm - self.ANN_X_MIN_CM)
                                  / self.ann_grid_cell_size)
        self.ann_grid_height = int((self.y_max_cm - self.ANN_Y_MIN_CM)
                                   / self.ann_grid_cell_size)

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

        # optional metric-centre cache
        self._center_overrides = {}
        self._load_center_cache()
        self._check_position_ids()

        print(f"  MmCows: RUNTIME grid = {self.grid_width}x"
              f"{self.grid_height} @ {self.grid_cell_size} cm "
              f"(worldcoord_from_worldgrid W[0,0]="
              f"{self.worldcoord_from_worldgrid_mat[0, 0]:.1f})")

    def _parse_frame_key(self, fname):
        """Return the frame key for an annotation/image filename.

        Supports two naming schemes:
          * legacy 8-digit zero-padded names  '00000000.json'      -> int(0)
          * timestamp names '1690309215_13-20-15.json'  -> '1690309215_13-20-15'
        The timestamp string itself is used as the frame key.
        """
        stem = os.path.splitext(fname)[0]
        if '_' in stem:  # timestamp <unix>_<HH-MM-SS>
            return stem
        return int(stem)  # legacy numeric frame

    def _frame_sort_key(self, key):
        """Sort key that orders both int (legacy) and str (timestamp) frames."""
        if isinstance(key, int):
            return (0, key)
        try:
            return (1, int(key.split('_')[0]))  # sort by leading unix timestamp
        except ValueError:
            return (1, 0)

    def _count_frames(self):
        """Frame list from annotations_positions/, or — for a sequence
        with NO ground truth — from the image files themselves.

        Sets self.has_gt: False marks the sequence as inference-only.
        """
        ann_dir = os.path.join(self.root, 'annotations_positions')
        frame_files = ([f for f in os.listdir(ann_dir) if f.endswith('.json')]
                       if os.path.isdir(ann_dir) else [])
        if frame_files:
            self.has_gt = True
            self.frame_list = sorted(
                (self._parse_frame_key(f) for f in frame_files),
                key=self._frame_sort_key,
            )
            return len(self.frame_list)

        # ── inference-only sequence: no annotations on disk ──
        self.has_gt = False
        self.frame_list = self._frame_list_from_images()
        print(f"  MmCows: ⚠ NO annotations in {ann_dir} — this sequence is "
              f"INFERENCE-ONLY ({len(self.frame_list)} frames derived from "
              f"Image_subsets/). It is excluded from every metric; "
              f"predictions are still exported.")
        return len(self.frame_list)

    def _frame_list_from_images(self):
        """Sorted frame keys present in EVERY camera's Image_subsets dir.

        Intersection, not union: PedestrianDataset.get_image_data() opens
        self.img_fpaths[cam][frame] for every camera, so a frame missing
        from one camera would raise KeyError at fetch time.
        """
        img_root = os.path.join(self.root, 'Image_subsets')
        cam_names = ['C1', 'C2', 'C3', 'C4'][:self.num_cam]
        per_cam = []
        for cam_name in cam_names:
            cam_dir = os.path.join(img_root, cam_name)
            if not os.path.isdir(cam_dir):
                raise FileNotFoundError(
                    f"{self.root}: no annotations AND no camera directory "
                    f"{cam_dir} — there is nothing to run inference on."
                )
            per_cam.append({self._parse_frame_key(f)
                            for f in os.listdir(cam_dir)
                            if f.endswith(('.jpg', '.png'))})
        common = set.intersection(*per_cam) if per_cam else set()
        if not common:
            raise ValueError(
                f"{self.root}: no annotations and no shared image frames "
                f"across {cam_names}."
            )
        dropped = set.union(*per_cam) - common
        if dropped:
            print(f"  MmCows: ⚠ {len(dropped)} frame(s) not present in ALL "
                  f"cameras — dropped from the inference frame list.")
        return sorted(common, key=self._frame_sort_key)

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
                frame = self._parse_frame_key(fname)
                if frame in frame_range:  # frame_range may be a set or a range
                    img_fpaths[cam_idx][frame] = os.path.join(cam_dir, fname)
        return img_fpaths

    def get_worldcoord_from_pos(self, pos):
        """Decode a packed positionID into metric world coordinates (cm).

        Uses the ANNOTATION grid only, so the result never changes when
        grid_cell_size / bounds / resolution are changed.
        """
        pos = int(pos)
        if self.ANN_STRIDE_AXIS == 'height':
            gx, gy = pos // self.ann_grid_height, pos % self.ann_grid_height
        else:
            gy, gx = pos // self.ann_grid_width, pos % self.ann_grid_width
        half = 0.5 if self.ANN_POS_CELL_CENTER else 0.0
        x_cm = self.ANN_X_MIN_CM + (gx + half) * self.ann_grid_cell_size
        y_cm = self.ANN_Y_MIN_CM + (gy + half) * self.ann_grid_cell_size
        return np.array([x_cm, y_cm], dtype=np.float32)

    def get_worldgrid_from_pos(self, pos):
        """positionID -> RUNTIME world-grid coordinates (float, sub-cell)."""
        return self.get_worldgrid_from_center(self.get_worldcoord_from_pos(pos))

    def get_worldgrid_from_center(self, center_xy):
        """Metric centre [x_cm, y_cm] -> world-grid (grid_x, grid_y).

        Exact inverse of worldcoord_from_worldgrid_mat.  (Moved here from
        MmCows3D so that the 2D path can use the very same transform.)
        """
        x_cm, y_cm = float(center_xy[0]), float(center_xy[1])
        grid_x = (x_cm - self.x_min_cm) / self.grid_cell_size
        grid_y = (y_cm - self.y_min_cm) / self.grid_cell_size
        return np.array([grid_x, grid_y], dtype=np.float32)

    def get_3d_boxes(self, frame_key):
        """Full 3D boxes (yaw/size/posture). Empty for a 2D-only dataset."""
        return {}

    def get_center_overrides(self, frame_key):
        """{cow_id: [x_cm, y_cm]} metric body centres, or {}.

        For a 3D dataset these are the 3D box centroids; for a 2D-only
        dataset they come from an optional centers_3d.json cache so that
        a 2D run can be supervised at EXACTLY the same points as the 3D run.
        """
        return self._center_overrides.get(frame_key, {})

    # ------------------------------------------------------------------
    @staticmethod
    def _cache_key(k):
        """Normalise a JSON frame key back to this dataset's key type.

        centers_3d.json stores every key as a STRING. Legacy numeric
        frames must come back as int ('00000000' -> 0); timestamp frames
        stay strings ('1690309215_13-20-15'). Mirrors _parse_frame_key().
        """
        k = str(k)
        try:
            return int(k)
        except ValueError:
            return k

    def _load_center_cache(self):
        """Populate self._center_overrides from centers_3d.json.

        {frame_key: {cow_id(int): [x_cm, y_cm]}}

        This is what makes a 2D-only run supervise the BEV heat-map at
        EXACTLY the same points as the 3D run. If it stays empty the
        centres silently fall back to the positionID annotation point.
        """
        self._center_overrides = {}
        self._center_cache_path = None

        # MmCows3D sets this to False for annotation_mode='2d_strict'.
        if not getattr(self, '_use_center_cache', True):
            return

        path = os.path.join(self.root, 'centers_3d.json')
        if not os.path.isfile(path):
            return

        try:
            with open(path) as f:
                raw = json.load(f)
        except (OSError, ValueError) as e:
            print(f"  MmCows: ⚠ could not read {path} ({e}); metric "
                  f"centres would fall back to positionID.")
            return

        n_pts = 0
        for frame_str, cows in (raw or {}).items():
            per_frame = {}
            for cid, xy in (cows or {}).items():
                try:
                    per_frame[int(cid)] = [float(xy[0]), float(xy[1])]
                except (TypeError, ValueError, IndexError):
                    continue
            if per_frame:
                self._center_overrides[self._cache_key(frame_str)] = per_frame
                n_pts += len(per_frame)

        self._center_cache_path = path

        known = list(getattr(self, 'frame_list', []) or [])
        if known:
            hit = len(set(self._center_overrides) & set(known))
            cover = f" | covers {hit}/{len(known)} frames of this sequence"
            if hit == 0:
                cover += "  ⚠ KEY MISMATCH (int vs timestamp?)"
        else:
            cover = ""

        print(f"  MmCows: metric-centre cache loaded: "
              f"{len(self._center_overrides)} frames / {n_pts} centres"
              f"{cover} <- {path}")

    @property
    def has_center_cache(self):
        return bool(self._center_overrides)

    # ------------------------------------------------------------------
    def _check_position_ids(self, max_files=50):
        """Loud sanity check on the assumed annotation grid."""
        ann_dir = os.path.join(self.root, 'annotations_positions')
        if not os.path.isdir(ann_dir):
            return  # inference-only sequence — nothing to check
        limit = self.ann_grid_width * self.ann_grid_height
        worst, n_bad, n_tot = -1, 0, 0
        for fname in sorted(os.listdir(ann_dir))[:max_files]:
            if not fname.endswith('.json'):
                continue
            with open(os.path.join(ann_dir, fname)) as f:
                for ped in json.load(f):
                    p = int(ped['positionID'])
                    n_tot += 1
                    worst = max(worst, p)
                    n_bad += int(p >= limit)
        if (self.ann_grid_cell_size != self.grid_cell_size) or n_bad:
            print(f"  MmCows: annotation grid = {self.ann_grid_width}x"
                  f"{self.ann_grid_height} @ {self.ann_grid_cell_size} cm "
                  f"| runtime grid = {self.grid_width}x{self.grid_height} @ "
                  f"{self.grid_cell_size} cm  (decoupled: OK)")
        if n_bad:
            print(f"  MmCows: ⚠ {n_bad}/{n_tot} positionIDs >= "
                  f"{limit} (max={worst}). ANN_GRID_CELL_SIZE / "
                  f"ANN_STRIDE_AXIS are WRONG -> run "
                  f"MmCows3D.infer_position_encoding() on a 3D sequence.")

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
