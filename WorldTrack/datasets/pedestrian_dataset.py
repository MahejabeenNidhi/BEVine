# WorldTrack/datasets/pedestrian_dataset.py
import os
import json
import random
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
        center_from_3d=True,
        # motion-aware stationary-frame filtering (train only)
        drop_stationary=False,
        stationary_min_disp_cm=5.0,
        stationary_keep_prob=0.0,
        stationary_max_run=5,
        # max |prev-cur| tracking offset in BEV cells; pairs beyond this
        # get NO tracking supervision. 15 = 150 cm at 10 cm/cell (mmcows).
        # MultiviewC p95 displacement is 790 cm
        track_offset_guard_cells: float = 15.0,
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

        # Stationary-frame filtering is a TRAIN-TIME augmentation only:
        # val/test frame sets must stay identical across experiments.
        self.drop_stationary = bool(drop_stationary) and bool(is_train)
        self.stationary_min_disp_cm = float(stationary_min_disp_cm)
        self.stationary_keep_prob = float(stationary_keep_prob)
        self.stationary_max_run = int(stationary_max_run)
        self.track_offset_guard_cells = float(track_offset_guard_cells)
        self.stationary_report = None

        self.data_aug_conf = {
            'final_dim': final_dim,
            'resize_lim': resize_lim,
        }
        self.kernel_size = 1.5
        self.max_objects = 60
        self.img_downsample = 4
        # 3D attributes (yaw/size/posture) are painted on a small disc
        # around the centre so that decoding at a peak that is 1-2 cells
        # off the exact GT centre still reads a supervised value.
        # attr_radius=2 -> +/-2 cells = +/-20 cm at grid_cell_size=10
        self.attr_radius = 2
        self.center_from_3d = center_from_3d
        _has_centers = (bool(getattr(base, 'has_3d', False))
                        or bool(getattr(base, '_center_overrides', {})))
        _has_centers = (bool(getattr(base, 'has_3d', False))
                        or bool(getattr(base, '_center_overrides', {})))
        if (self.center_from_3d and not _has_centers
                and getattr(base, 'has_gt', True)):
            mode = getattr(base, 'annotation_mode', 'n/a')
            print(f"[GT] WARNING: center_from_3d=True but '{base.root}' provides "
                  f"no metric centres (annotation_mode={mode!r}; no "
                  f"3D_annotations/, no centers_3d.json).\n"
                  f"     BEV centres fall back to the positionID annotation "
                  f"point, which sits ~35 cm from the body centre.\n"
                  f"     -> This GT is NOT comparable with a 3D run.")
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
        _cm_grid = float(getattr(base, 'grid_cell_size', 0) or 0)
        self.cm_per_mem_cell = (
            _cm_grid * float(self.vox_util.default_vox_size_X)
            if _cm_grid > 0 else None)

        if hasattr(self.base, 'frame_list'):
            # MmCows / MmCows3D: split by position in the sorted key list so
            # this works for both int (legacy) and str (timestamp) frame keys.
            all_keys = list(self.base.frame_list)
            n = len(all_keys)
            if self.use_all_frames:
                frame_range = set(all_keys)
            elif self.is_train:
                frame_range = set(all_keys[:int(n * 0.9)])
            else:
                frame_range = set(all_keys[int(n * 0.9):])
        else:
            # Wildtrack / MultiviewX: unchanged integer-range behaviour.
            if self.use_all_frames:
                frame_range = range(0, self.num_frame)
            elif self.is_train:
                frame_range = range(0, int(self.num_frame * 0.9))
            else:
                frame_range = range(int(self.num_frame * 0.9), self.num_frame)

        # drop frames where NO animal moved, before anything is
        # loaded. Runs only on MmCows-style bases (metric centres /
        # positionID decoding) and only on the train split.
        if self.drop_stationary:
            if (hasattr(self.base, 'frame_list')
                    and hasattr(self.base, 'get_worldcoord_from_pos')):
                frame_range, self.stationary_report = (
                    self._filter_stationary_frames(frame_range)
                )
            else:
                print("  [STATIONARY] ⚠ base dataset exposes neither "
                      "frame_list nor get_worldcoord_from_pos — skipping "
                      "stationary-frame filtering.")

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        self.download(frame_range)

        # Canonical, deterministically-ordered list of the frame keys that
        # actually made it into this split. __getitem__ indexes into this.
        if hasattr(self.base, '_frame_sort_key'):
            self.frame_keys = sorted(
                self.world_gt.keys(), key=self.base._frame_sort_key
            )
        else:
            self.frame_keys = sorted(self.world_gt.keys())

        self._report_center_coverage()

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
        if not os.path.isdir(ann_dir):
            return  # inference-only sequence — no gt.txt to write
        has_parser = hasattr(self.base, '_parse_frame_key')
        key_to_idx = (
            {k: i for i, k in enumerate(self.base.frame_list)}
            if has_parser else {}
        )
        for fname in sorted(os.listdir(ann_dir)):
            if has_parser:
                frame = key_to_idx.get(self.base._parse_frame_key(fname), -1)
            else:
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
                _ctr = (self.base.get_center_overrides(
                    self.base._parse_frame_key(fname))
                        if (self.center_from_3d and has_parser) else {})
                _c = _ctr.get(int(single_pedestrian['personID']))
                if _c is not None:
                    grid_x, grid_y = self.base.get_worldgrid_from_center(_c)
                else:
                    grid_x, grid_y = self.base.get_worldgrid_from_pos(
                        single_pedestrian['positionID'])
                og_gt.append(np.array([frame, int(round(float(grid_x))), int(round(float(grid_y)))]))
        if not og_gt:
            return  # annotation dir exists but held no usable rows
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0
        ann_dir = os.path.join(self.root, 'annotations_positions')
        if not os.path.isdir(ann_dir):
            # ── inference-only sequence: no GT at all ──
            # Shapes matter: (0, 2) / (0, 4), NOT the (0,) that
            # torch.tensor([]) produces — __getitem__ indexes [:, 0:1].
            for frame in frame_range:
                self.world_gt[frame] = (
                    torch.zeros((0, 2), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.float32),
                )
                self.imgs_gt[frame] = {
                    cam: (torch.zeros((0, 4), dtype=torch.float32),
                          torch.zeros((0,), dtype=torch.float32))
                    for cam in range(self.num_cam)
                }
            print(f"  PedestrianDataset: "
                  f"{os.path.basename(str(self.root).rstrip('/'))} has no "
                  f"annotations — {len(frame_range)} frames registered as "
                  f"inference-only (zero GT rows).")
            return
        for fname in sorted(os.listdir(ann_dir)):
            frame = (self.base._parse_frame_key(fname)
                     if hasattr(self.base, '_parse_frame_key')
                     else int(fname.split('.')[0]))
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
                    (torch.tensor(world_pts, dtype=torch.float32)
                     if world_pts
                     else torch.zeros((0, 2), dtype=torch.float32)),
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
    # Motion-aware stationary-frame filtering (train-time augmentation)
    # ------------------------------------------------------------------
    def _load_annotations_by_frame(self):
        """frame_key -> raw annotation list (positionID fallback source)."""
        ann_dir = os.path.join(self.root, 'annotations_positions')
        out = {}
        for fname in sorted(os.listdir(ann_dir)):
            if not fname.endswith('.json'):
                continue
            key = (self.base._parse_frame_key(fname)
                   if hasattr(self.base, '_parse_frame_key')
                   else int(fname.split('.')[0]))
            with open(os.path.join(ann_dir, fname)) as f:
                out[key] = json.load(f)
        return out

    def _filter_stationary_frames(self, frame_range):
        """Drop frames in which NO animal moved more than
        ``stationary_min_disp_cm`` w.r.t. the immediately preceding frame.

        A frame is KEPT whenever even a single shared cow ID moved more
        than the threshold — only all-stationary frames are candidates.
        Displacement is measured in metric cm from the 3D centre cache
        (sub-cm precision); frames without a cached centre fall back to
        the 10 cm-quantised positionID decode and are counted.

        Returns (filtered_frame_range_set, report_dict).
        """
        min_disp = self.stationary_min_disp_cm
        keep_prob = self.stationary_keep_prob
        max_run = self.stationary_max_run
        rng = random.Random(20240717)   # fixed seed -> reproducible filtering

        keys = list(self.base.frame_list)
        centres, n_cache = {}, 0
        for k in keys:
            c = self.base.get_center_overrides(k)
            if c:
                centres[k] = {int(i): (float(v[0]), float(v[1]))
                              for i, v in c.items()}
                n_cache += 1
        missing = [k for k in keys if k not in centres]
        n_posid = len(missing)
        if missing:
            ann_by_frame = self._load_annotations_by_frame()
            for k in missing:
                pts = {}
                for ped in ann_by_frame.get(k, []):
                    xy = self.base.get_worldcoord_from_pos(ped['positionID'])
                    pts[int(ped['personID'])] = (float(xy[0]), float(xy[1]))
                centres[k] = pts

        kept, dropped = [], []
        run = 0                     # consecutive dropped stationary frames
        n_stationary = 0
        n_split = 0
        for i, k in enumerate(keys):
            if k not in frame_range:
                continue
            n_split += 1
            if i == 0:
                kept.append(k)
                continue
            prev, cur = centres.get(keys[i - 1], {}), centres.get(k, {})
            shared = set(prev) & set(cur)
            max_d = max(
                (float(np.hypot(cur[p][0] - prev[p][0],
                                cur[p][1] - prev[p][1])) for p in shared),
                default=None,
            )
            # Moved (or cannot be proven stationary) -> ALWAYS keep.
            if max_d is None or max_d > min_disp:
                kept.append(k)
                run = 0
                continue
            n_stationary += 1
            if run < max_run and rng.random() >= keep_prob:
                dropped.append(k)
                run += 1
            else:
                kept.append(k)
                run = 0

        report = {
            'sequence': str(self.root),
            'frames_in_split': n_split,
            'stationary': n_stationary,
            'dropped': len(dropped),
            'kept': len(kept),
            'min_disp_cm': min_disp,
            'keep_prob': keep_prob,
            'max_drop_run': max_run,
            'metric_centre_frames': n_cache,
            'positionID_fallback_frames': n_posid,
            'dropped_keys': [str(k) for k in dropped],
        }
        seq = os.path.basename(str(self.root).rstrip('/'))
        print(f"  [STATIONARY] {seq}: {n_stationary}/{n_split} frames fully "
              f"stationary (no cow moved > {min_disp} cm) -> dropped "
              f"{len(dropped)}, kept {len(kept)} "
              f"(metric centres: {n_cache}, positionID fallback: {n_posid})")
        if n_posid:
            print(f"  [STATIONARY] ⚠ {n_posid} frames measured from "
                  f"positionID (10 cm quantised): a cow moving <10 cm counts "
                  f"as stationary there. Run MmCows3D.export_center_cache() "
                  f"if you need sub-cell sensitivity on every frame.")
        return set(kept), report

    # ------------------------------------------------------------------
    # BEV / image GT helpers
    # ------------------------------------------------------------------
    def get_bev_gt(self, mem_pts, mem_pts_prev, pids, pids_pre,
                   mem_pts_3d=None, ids_3d=None, boxes_3d=None,
                   mem_pts_3d_prev=None, ids_3d_prev=None):
        center = torch.zeros((1, self.Y, self.X), dtype=torch.float32)
        valid_mask = torch.zeros((1, self.Y, self.X), dtype=torch.bool)
        offset = torch.zeros((4, self.Y, self.X), dtype=torch.float32)
        person_ids = torch.zeros((1, self.Y, self.X), dtype=torch.long)

        yaw_bev = torch.zeros((2, self.Y, self.X), dtype=torch.float32)
        size_bev = torch.zeros((3, self.Y, self.X), dtype=torch.float32)
        posture_bev = torch.zeros((1, self.Y, self.X), dtype=torch.float32)
        valid_3d = torch.zeros((1, self.Y, self.X), dtype=torch.bool)

        prev_pts = dict(zip(pids_pre.int().tolist(), mem_pts_prev[0]))

        # ── 3D centre lookups (current and previous frame) ──
        use3d = bool(self.center_from_3d and mem_pts_3d is not None and ids_3d)
        c3d = ({int(i): mem_pts_3d[0][k] for k, i in enumerate(ids_3d)}
               if use3d else {})
        c3d_prev = ({int(i): mem_pts_3d_prev[0][k]
                     for k, i in enumerate(ids_3d_prev)}
                    if (use3d and mem_pts_3d_prev is not None and ids_3d_prev)
                    else {})

        pid_to_px = {}

        for pts, pid in zip(mem_pts[0], pids):
            pid_i = int(pid.item())
            # CENTRE SOURCE: 3D centroid when available, else the legacy
            # 2D annotation position. Both are already augmented.
            src = c3d.get(pid_i)
            ct = (src[:2] if src is not None else pts[:2])
            ct_int = ct.int()
            if (ct_int[0] < 0 or ct_int[0] >= self.X
                    or ct_int[1] < 0 or ct_int[1] >= self.Y):
                continue
            # centre heat-map: oriented, footprint-sized Gaussian when a
            # 3D box exists for this animal, legacy isotropic otherwise.
            box = boxes_3d.get(pid_i) if boxes_3d else None
            drew = False
            if (box is not None and self.cm_per_mem_cell
                    and float(box.get('length', 0)) > 0
                    and float(box.get('width', 0)) > 0):
                sigma_l = max(float(box['length'])
                              / self.cm_per_mem_cell / 6.0, 1.5)
                sigma_w = max(float(box['width'])
                              / self.cm_per_mem_cell / 6.0, 1.5)
                for c in center:
                    basic.draw_oriented_gaussian(
                        c, ct_int, sigma_l, sigma_w,
                        float(box.get('yaw', 0.0)))
                drew = True
            if not drew:
                for c in center:
                    basic.draw_umich_gaussian(c, ct_int, self.kernel_size)
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:2, ct_int[1], ct_int[0]] = ct - ct_int
            person_ids[:, ct_int[1], ct_int[0]] = pid
            pid_to_px[pid_i] = (int(ct_int[0]), int(ct_int[1]))

            # TRACKING OFFSET: must use the SAME centre convention in
            # both frames, otherwise the 2D->3D offset leaks into it.
            prev_xy = None
            if src is not None and pid_i in c3d_prev:
                prev_xy = c3d_prev[pid_i][:2]
            elif src is None and pid in pids_pre:
                prev_xy = prev_pts[pid_i][:2]
            if prev_xy is not None:
                t_off = prev_xy - ct_int
                if t_off.abs().max() > self.track_offset_guard_cells:
                    continue
                offset[2:, ct_int[1], ct_int[0]] = t_off

        # 3D attributes: painted at the SAME pixel as the centre
        if mem_pts_3d is not None and ids_3d and boxes_3d:
            r = self.attr_radius
            # Nearest-centre arbitration: a cell belongs to the animal
            # whose centre is CLOSEST, not to whichever was iterated last.
            # Also circular instead of square, so no cell more than r
            # cells away from a centre is ever supervised.
            owner_d = torch.full((self.Y, self.X), float('inf'))
            _yy = torch.arange(self.Y, dtype=torch.float32).view(-1, 1)
            _xx = torch.arange(self.X, dtype=torch.float32).view(1, -1)
            for pts, cow_id in zip(mem_pts_3d[0], ids_3d):
                cid = int(cow_id)
                box = boxes_3d.get(cid)
                if box is None:
                    continue
                if cid in pid_to_px:
                    cx, cy = pid_to_px[cid]
                else:
                    ct_int = pts[:2].int()
                    cx, cy = int(ct_int[0]), int(ct_int[1])
                if not (0 <= cx < self.X and 0 <= cy < self.Y):
                    continue

                x0, x1 = max(cx - r, 0), min(cx + r + 1, self.X)
                y0, y1 = max(cy - r, 0), min(cy + r + 1, self.Y)
                d = ((_yy[y0:y1] - cy) ** 2 + (_xx[:, x0:x1] - cx) ** 2).sqrt()
                m = (d <= float(r)) & (d < owner_d[y0:y1, x0:x1])
                if not bool(m.any()):
                    continue
                owner_d[y0:y1, x0:x1] = torch.where(
                    m, d, owner_d[y0:y1, x0:x1])

                yaw_val = float(box.get('yaw', 0.0))
                # Doubled-angle representation: supervise (sin 2θ, cos 2θ),
                # which is invariant to θ -> θ + π, matching the
                # pi-periodicity of an oriented box.
                sin_v = float(np.sin(2.0 * yaw_val))
                cos_v = float(np.cos(2.0 * yaw_val))
                l = float(box.get('length', 0.0)) / basic.SIZE_SCALE_CM
                w = float(box.get('width', 0.0)) / basic.SIZE_SCALE_CM
                h = float(box.get('height', 0.0)) / basic.SIZE_SCALE_CM
                post = (1.0 if str(box.get('posture', '')).lower() == 'standing'
                        else 0.0)

                yaw_bev[0, y0:y1, x0:x1][m] = sin_v
                yaw_bev[1, y0:y1, x0:x1][m] = cos_v
                size_bev[0, y0:y1, x0:x1][m] = l
                size_bev[1, y0:y1, x0:x1][m] = w
                size_bev[2, y0:y1, x0:x1][m] = h
                posture_bev[0, y0:y1, x0:x1][m] = post
                valid_3d[0, y0:y1, x0:x1][m] = True

        return (center, valid_mask, person_ids, offset,
                yaw_bev, size_bev, posture_bev, valid_3d)

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
    def _report_center_coverage(self):
        """State explicitly WHERE every BEV centre of this split comes from.

        A 3D run and a 2D-only run are only comparable if this report is
        IDENTICAL for both. Any non-zero 'fall back to positionID' count
        is a ~35 cm systematic shift of that animal's heat-map peak.
        """
        seq = os.path.basename(str(self.base.root).rstrip('/'))
        mode = getattr(self.base, 'annotation_mode', 'n/a')
        if not getattr(self.base, 'has_gt', True):
            print(f"[GT] {seq}: no annotations — inference-only sequence.")
            return
        if not self.center_from_3d:
            print(f"[GT] {seq} ({mode!r}): center_from_3d=False -> ALL BEV "
                  f"centres come from positionID.")
            return

        n_ann, n_wo, n_frames_wo = 0, 0, 0
        for f in self.frame_keys:
            centers = self.base.get_center_overrides(f)
            if not centers:
                n_frames_wo += 1
            pids = self.world_gt[f][1].int().tolist()
            n_ann += len(pids)
            n_wo += sum(1 for p in pids if int(p) not in centers)

        print(f"[GT] {seq} ({mode!r}): metric centres for "
              f"{n_ann - n_wo}/{n_ann} annotations, "
              f"{len(self.frame_keys) - n_frames_wo}/{len(self.frame_keys)} "
              f"frames covered.")
        if n_wo:
            print(f"[GT]  ⚠ {n_wo} annotation(s) fall back to the positionID "
                  f"point (~35 cm off the body centre). The 3D and 2D runs "
                  f"only match if this number is IDENTICAL in both.")

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
            if getattr(self.base, 'IMG_HFLIP', False):        # C21
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
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
        frame = self.frame_keys[index]
        pre_frame = self.frame_keys[max(index - 1, 0)]
        frame_id = index if isinstance(frame, str) else frame // self.base.frame_step
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

        # ── metric CENTRE source: identical in 2D-only and 3D runs ──
        # get_3d_boxes()        -> full boxes (yaw/size/posture) or {}
        # get_center_overrides()-> {cow_id: [x_cm, y_cm]} (3D centroid or cache)
        boxes_3d = self.base.get_3d_boxes(frame)
        boxes_3d_prev = self.base.get_3d_boxes(pre_frame)
        centers_cm = self.base.get_center_overrides(frame)
        centers_cm_prev = self.base.get_center_overrides(pre_frame)

        def _lift(centers):
            if not centers:
                return [], None
            ids = sorted(centers.keys())
            g = np.stack([self.base.get_worldgrid_from_center(centers[i])
                          for i in ids], axis=0)
            return ids, torch.cat((
                torch.tensor(g, dtype=torch.float32),
                torch.zeros((len(ids), 1), dtype=torch.float32),
            ), dim=1).unsqueeze(0)

        ids_3d, worldgrid_pts_3d = _lift(centers_cm)
        ids_3d_prev, worldgrid_pts_3d_prev = _lift(centers_cm_prev)

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
            if worldgrid_pts_3d is not None:
                worldgrid_pts_3d = geom.apply_4x4(
                    augment.unsqueeze(0), worldgrid_pts_3d
                )
            if worldgrid_pts_3d_prev is not None:
                worldgrid_pts_3d_prev = geom.apply_4x4(
                    augment.unsqueeze(0), worldgrid_pts_3d_prev
                )

        mem_pts = self.vox_util.Ref2Mem(
            worldgrid_pts, self.Y, self.Z, self.X
        )
        mem_pts_pre = self.vox_util.Ref2Mem(
            worldgrid_pts_pre, self.Y, self.Z, self.X
        )
        mem_pts_3d = (
            self.vox_util.Ref2Mem(worldgrid_pts_3d, self.Y, self.Z, self.X)
            if worldgrid_pts_3d is not None else None
        )
        mem_pts_3d_prev = (
            self.vox_util.Ref2Mem(worldgrid_pts_3d_prev, self.Y, self.Z, self.X)
            if worldgrid_pts_3d_prev is not None else None
        )

        (center_bev, valid_bev, pid_bev, offset_bev,
         yaw_bev, size_bev, posture_bev, valid_3d) = self.get_bev_gt(
            mem_pts, mem_pts_pre, world_pids, world_pid_pre,
            mem_pts_3d=mem_pts_3d, ids_3d=ids_3d, boxes_3d=boxes_3d,
            mem_pts_3d_prev=mem_pts_3d_prev, ids_3d_prev=ids_3d_prev,
        )

        grid_gt_3d = torch.zeros((self.max_objects, 11), dtype=torch.float32)
        if mem_pts_3d is not None and ids_3d and boxes_3d:
            pid2mem = {int(p.item()): mem_pts[0, i, :2]
                       for i, p in enumerate(world_pids)}
            k = 0
            for idx, cid in enumerate(ids_3d):
                if k >= self.max_objects:
                    break
                box = boxes_3d.get(int(cid))
                if box is None:
                    continue  # centre-only entry -> no 3D box GT
                center = box['center']
                src = (mem_pts_3d[0, idx, :2] if self.center_from_3d
                       else pid2mem.get(int(cid), mem_pts_3d[0, idx, :2]))
                grid_gt_3d[k, 0] = src[0]
                grid_gt_3d[k, 1] = src[1]
                grid_gt_3d[k, 2] = float(center[0])
                grid_gt_3d[k, 3] = float(center[1])
                grid_gt_3d[k, 4] = 0.0
                grid_gt_3d[k, 5] = float(cid)
                grid_gt_3d[k, 6] = float(box.get('yaw', 0.0))
                grid_gt_3d[k, 7] = float(box.get('length', 0.0))
                grid_gt_3d[k, 8] = float(box.get('width', 0.0))
                grid_gt_3d[k, 9] = float(box.get('height', 0.0))
                grid_gt_3d[k, 10] = (
                    1.0 if str(box.get('posture', '')).lower() == 'standing'
                    else 0.0
                )
                k += 1

        grid_gt = torch.zeros((self.max_objects, 3), dtype=torch.long)
        n_pts = worldgrid_pts.shape[1]
        # round(): a plain long-cast truncates toward zero (up to -1 cell bias)
        grid_gt[:n_pts, :2] = torch.round(worldgrid_pts_org).long()
        grid_gt[:n_pts, 2] = world_pids

        if self.center_from_3d and centers_cm:
            c = {int(i): torch.round(torch.tensor(
                self.base.get_worldgrid_from_center(centers_cm[i]),
                dtype=torch.float32)).long()
                 for i in ids_3d}
            for n in range(min(n_pts, self.max_objects)):
                cow_id = int(grid_gt[n, 2].item())
                if cow_id in c:
                    grid_gt[n, :2] = c[cow_id]

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
            'frame': frame_id,
            'sequence_num': int(self.sequence_num),
            'grid_gt': grid_gt,
            'intrinsic_original': self.calibration[
                'intrinsic'
            ].clone(),
            'img_gt_2d': img_gt_2d,
            'img_paths': img_paths_str,
            # BASE (non-augmented) grid→world 3×3 matrix for reprojection loss
            'worldcoord_from_worldgrid': worldcoord_from_worldgrid2d,
            # NEW (Phase 4): compact 3D-box GT (mem centre + world centre + id)
            'grid_gt_3d': grid_gt_3d,
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
            # ── NEW 3D targets ──
            'yaw_bev': yaw_bev,
            'size_bev': size_bev,
            'posture_bev': posture_bev,
            'valid_3d': valid_3d,
        }
        return item, target
