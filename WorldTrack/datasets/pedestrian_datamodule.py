# WorldTrack/datasets/pedestrian_datamodule.py

import os
import json
import numpy as np
from typing import Optional

import lightning as pl
from torch.utils.data import DataLoader

from datasets.multiviewx_dataset import MultiviewX
from datasets.wildtrack_dataset import Wildtrack
from datasets.mmcows_dataset import MmCows
from datasets.mmcows3d_dataset import MmCows3D
from datasets.multiviewc_dataset import MultiviewC
from datasets.jerccows_dataset import JerCCows
from datasets.pedestrian_dataset import PedestrianDataset
from datasets.multiseq_pedestrian_dataset import MultiSeqPedestrianDataset
from datasets.sampler import TemporalSampler, MultiSeqTemporalSampler


class PedestrianDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "../data/MultiviewX",
        batch_size: int = 2,
        num_workers: int = 4,
        resolution=None,
        bounds=None,
        accumulate_grad_batches=8,
        # multi-sequence knobs
        multi_seq: bool = False,
        manifest_name: str = 'sequences_mmCows_all.json',
        center_from_3d: bool = True,
        # '3d' | '2d' | '2d_strict'  — see MmCows3D.
        # Switches 3D supervision off WITHOUT deleting 3D_annotations/.
        annotation_mode: str = '3d',
        require_gt_parity: bool = False,
        # stationary-frame filtering (train split only)
        drop_stationary: bool = False,
        stationary_min_disp_cm: float = 5.0,
        stationary_keep_prob: float = 0.0,
        stationary_max_run: int = 5,
        track_offset_guard_cells: float = 15.0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.bounds = bounds
        self.accumulate_grad_batches = accumulate_grad_batches
        self.multi_seq = multi_seq
        self.manifest_name = manifest_name
        self.center_from_3d = center_from_3d
        self.annotation_mode = str(annotation_mode).lower()
        self.require_gt_parity = bool(require_gt_parity)
        self.drop_stationary = bool(drop_stationary)
        self.stationary_min_disp_cm = float(stationary_min_disp_cm)
        self.stationary_keep_prob = float(stationary_keep_prob)
        self.stationary_max_run = int(stationary_max_run)
        self.track_offset_guard_cells = float(track_offset_guard_cells)

        self.dataset = os.path.basename(self.data_dir)

        self.data_predict = None
        self.data_test = None
        self.data_val = None
        self.data_train = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _create_base(data_dir: str, annotation_mode: str = '3d'):
        """Instantiate the right base dataset class for *data_dir*."""
        name = os.path.basename(data_dir).lower()
        if 'wildtrack' in name:
            return Wildtrack(data_dir)
        elif 'multiviewx' in name:
            return MultiviewX(data_dir)
        elif 'mmcows' in name:
            return MmCows3D(data_dir, annotation_mode=annotation_mode)
        elif 'multiviewc' in name:  # 'multiviewc_data' contains neither
            return MultiviewC(data_dir)  # 'multiviewx' nor 'mmcows' — safe
        elif 'jerccows' in name:
            return JerCCows(data_dir)
        else:
            raise ValueError(f'Unknown dataset name {name}')

    def _create_base_for_seq(self, seq_dir: str):
        """Infer the dataset type from the multi-sequence ROOT, because
        sequence folder names carry no dataset identifier."""
        parent_name = os.path.basename(self.data_dir).lower()
        if 'wildtrack' in parent_name:
            return Wildtrack(seq_dir)
        elif 'multiviewx' in parent_name:
            return MultiviewX(seq_dir)
        elif 'mmcows' in parent_name:
            return MmCows3D(seq_dir, annotation_mode=self.annotation_mode)
        elif 'multiviewc' in parent_name:
            return MultiviewC(seq_dir)
        elif 'jerccows' in parent_name:
            return JerCCows(seq_dir)
        else:
            raise ValueError(
                f'Cannot infer dataset type from parent directory: '
                f'{self.data_dir}'
            )

    def _create_multiseq_dataset(self, split: str, is_train: bool):
        """Build a MultiSeqPedestrianDataset for the requested split.

        Each sequence directory listed in the JSON manifest becomes its
        own PedestrianDataset with ``use_all_frames=True``.

        Sequence numbers are offset per split so that train / val / test
        ids never collide in the temporal cache:
            train  →  0 …  9 999
            val    → 10 000 … 19 999
            test   → 20 000 … 29 999
        """
        manifest_path = os.path.join(self.data_dir, self.manifest_name)
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        if split not in manifest:
            raise ValueError(
                f"Split '{split}' not in manifest. "
                f"Available: {list(manifest.keys())}"
            )

        seq_offset = {'train': 0, 'val': 10_000, 'test': 20_000}.get(
            split, 0
        )

        datasets = []
        for seq_idx, seq_name in enumerate(manifest[split]):
            seq_dir = os.path.join(self.data_dir, seq_name)
            if not os.path.exists(seq_dir):
                print(
                    f"WARNING: Sequence dir not found: {seq_dir}, "
                    f"skipping."
                )
                continue

            base = self._create_base_for_seq(seq_dir)
            if not getattr(base, 'has_gt', True):
                if split not in ('test', 'predict'):
                    raise ValueError(
                        f"Sequence '{seq_name}' (split '{split}') has NO "
                        f"ground-truth annotations (no usable "
                        f"annotations_positions/). Unannotated sequences "
                        f"are supported for TEST/PREDICT only: training on "
                        f"them would supervise an EMPTY heat-map (teaching "
                        f"the detector to predict nothing), and validation "
                        f"metrics would be undefined. Move '{seq_name}' to "
                        f"the manifest's 'test' list."
                    )
                print(f"  Seq {seq_offset + seq_idx}: {seq_name} has NO "
                      f"annotations -> inference-only (excluded from every "
                      f"metric; predictions go to *_unlabeled.txt)")
            ds = PedestrianDataset(
                base,
                is_train=is_train,
                resolution=self.resolution,
                bounds=self.bounds,
                use_all_frames=True,
                sequence_num=seq_offset + seq_idx,
                center_from_3d=self.center_from_3d,
                drop_stationary=self.drop_stationary,
                stationary_min_disp_cm=self.stationary_min_disp_cm,
                stationary_keep_prob=self.stationary_keep_prob,
                stationary_max_run=self.stationary_max_run,
                track_offset_guard_cells=self.track_offset_guard_cells,
            )
            print(
                f"  Seq {seq_offset + seq_idx}: {seq_name} "
                f"-> {len(ds)} frames"
            )
            datasets.append(ds)

        # stationary-filter summary + audit file
        reports = [getattr(ds, 'stationary_report', None) for ds in datasets]
        reports = [r for r in reports if r]
        if reports:
            tot_f = sum(r['frames_in_split'] for r in reports)
            tot_s = sum(r['stationary'] for r in reports)
            tot_d = sum(r['dropped'] for r in reports)
            print(f"  [STATIONARY] split '{split}': {tot_s}/{tot_f} frames "
                  f"fully stationary; dropped {tot_d} "
                  f"({100.0 * tot_d / max(tot_f, 1):.1f}% of split)")
            try:
                out_path = os.path.join(self.data_dir,
                                        f'stationary_report_{split}.json')
                with open(out_path, 'w') as f:
                    json.dump(reports, f, indent=1)
                print(f"  [STATIONARY] per-sequence report -> {out_path}")
            except OSError as e:
                print(f"  [STATIONARY] ⚠ could not write report ({e})")

        if not datasets:
            raise ValueError(
                f"No valid sequences for split '{split}'"
            )

        n_unl = sum(1 for d in datasets
                    if not getattr(d.base, 'has_gt', True))
        if n_unl:
            print(f"  Split '{split}': {n_unl}/{len(datasets)} sequence(s) "
                  f"are inference-only (no GT); the remaining "
                  f"{len(datasets) - n_unl} are evaluated normally.")

        return MultiSeqPedestrianDataset(datasets)

    def _get_sampler(self, dataset, shuffle_sequences: bool = False):
        """Return the right sampler for *dataset*."""
        if getattr(dataset, 'is_multi_seq', False):
            return MultiSeqTemporalSampler(
                dataset,
                batch_size=self.batch_size,
                accumulate_grad_batches=self.accumulate_grad_batches,
                shuffle_sequences=shuffle_sequences,
            )
        return TemporalSampler(
            dataset,
            batch_size=self.batch_size,
            accumulate_grad_batches=self.accumulate_grad_batches,
        )

    def get_base_extrinsics(self):
        """
        Returns (base_dict, cam_names) where:
          base_dict : dict {cam_name: ndarray (3,4)} of the base
                      (pre-refinement) extrinsic matrices loaded by the dataset.
          cam_names : list of str, ordered by camera index 0..S-1.

        Camera names are derived dynamically from the base dataset object.
        Only valid after setup() has been called.

        Raises RuntimeError if no dataset split has been set up yet.
        """
        ds = (self.data_train or self.data_val
              or self.data_test or self.data_predict)

        if ds is None:
            raise RuntimeError(
                "No dataset split has been set up yet. "
                "Call setup() first."
            )

        # In multi-seq mode, use the first sub-dataset
        if hasattr(ds, 'datasets') and ds.datasets:
            base = ds.datasets[0].base
        elif hasattr(ds, 'base'):
            base = ds.base
        else:
            raise RuntimeError(
                "Cannot access base dataset from datamodule."
            )

        # Camera names
        if hasattr(base, 'cam_names'):
            cam_names = list(base.cam_names)
        else:
            cam_names = [f'C{i + 1}' for i in range(base.num_cam)]

        # Build dict of (3, 4) extrinsic matrices
        base_dict = {}
        for i, name in enumerate(cam_names):
            ext = np.array(
                base.extrinsic_matrices[i], dtype=np.float32
            )
            if ext.shape == (4, 4):
                ext = ext[:3]
            base_dict[name] = ext

        return base_dict, cam_names

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):

        if self.multi_seq:
            # ── multi-sequence mode ──────────────────────────────
            if stage == 'fit':
                print("Setting up multi-seq TRAIN …")
                self.data_train = self._create_multiseq_dataset(
                    'train', is_train=True
                )
                print(f"  Total train frames: {len(self.data_train)}")

            if stage in ('fit', 'validate'):
                print("Setting up multi-seq VAL …")
                self.data_val = self._create_multiseq_dataset(
                    'val', is_train=False
                )
                print(f"  Total val frames: {len(self.data_val)}")

            if stage == 'test':
                print("Setting up multi-seq TEST …")
                self.data_test = self._create_multiseq_dataset(
                    'test', is_train=False
                )
                print(f"  Total test frames: {len(self.data_test)}")

            if stage == 'predict':
                self.data_predict = self._create_multiseq_dataset(
                    'test', is_train=False
                )

        else:
            # single-sequence mode (backward compatible)
            base = self._create_base(self.data_dir)

            if stage == 'fit':
                self.data_train = PedestrianDataset(
                    base, is_train=True,
                    resolution=self.resolution, bounds=self.bounds, center_from_3d=self.center_from_3d,
                    drop_stationary=self.drop_stationary,
                    stationary_min_disp_cm=self.stationary_min_disp_cm,
                    stationary_keep_prob=self.stationary_keep_prob,
                    stationary_max_run=self.stationary_max_run,
                    track_offset_guard_cells=self.track_offset_guard_cells,
                )
            if stage in ('fit', 'validate'):
                self.data_val = PedestrianDataset(
                    base, is_train=False,
                    resolution=self.resolution, bounds=self.bounds, center_from_3d=self.center_from_3d,
                    drop_stationary=self.drop_stationary,
                    stationary_min_disp_cm=self.stationary_min_disp_cm,
                    stationary_keep_prob=self.stationary_keep_prob,
                    stationary_max_run=self.stationary_max_run,
                    track_offset_guard_cells=self.track_offset_guard_cells,
                )
            if stage == 'test':
                self.data_test = PedestrianDataset(
                    base, is_train=False,
                    resolution=self.resolution, bounds=self.bounds, center_from_3d=self.center_from_3d,
                    drop_stationary=self.drop_stationary,
                    stationary_min_disp_cm=self.stationary_min_disp_cm,
                    stationary_keep_prob=self.stationary_keep_prob,
                    stationary_max_run=self.stationary_max_run,
                    track_offset_guard_cells=self.track_offset_guard_cells,
                )
            if stage == 'predict':
                self.data_predict = PedestrianDataset(
                    base, is_train=False,
                    resolution=self.resolution, bounds=self.bounds, center_from_3d=self.center_from_3d,
                    drop_stationary=self.drop_stationary,
                    stationary_min_disp_cm=self.stationary_min_disp_cm,
                    stationary_keep_prob=self.stationary_keep_prob,
                    stationary_max_run=self.stationary_max_run,
                    track_offset_guard_cells=self.track_offset_guard_cells,
                )

    # ------------------------------------------------------------------
    # dataloaders
    # ------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self._get_sampler(
                self.data_train, shuffle_sequences=True
            ),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self._get_sampler(
                self.data_val, shuffle_sequences=False
            ),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,
            num_workers=1,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
