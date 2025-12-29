import os
import torch
from typing import List, Dict, Optional
import lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from datasets.multiviewx_dataset import MultiviewX
from datasets.wildtrack_dataset import Wildtrack
from datasets.pedestrian_dataset import PedestrianDataset
from datasets.tracktacular_mmcows_dataset import TrackTacularMMCows
from datasets.sampler import TemporalSampler

from datasets.multi_dataset_config import DatasetConfig, MultiDatasetConfig
from datasets.multi_dataset_sampler import MultiDatasetBatchSampler
from collections import defaultdict


def single_dataset_collate_fn(batch):
    """
    Custom collate function for single-dataset mode.
    Ensures consistent batch formatting with multi-dataset mode.
    """
    if len(batch) == 0:
        return None, None

    items = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack tensors
    collated_item = {}
    collated_target = {}

    for key in items[0].keys():
        if isinstance(items[0][key], torch.Tensor):
            collated_item[key] = torch.stack([item[key] for item in items])
        elif key in ['frame', 'sequence_num', 'has_gt', 'dataset_id']:
            # Convert to tensors for consistency
            if key == 'has_gt':
                collated_item[key] = torch.tensor([item[key] for item in items], dtype=torch.bool)
            else:
                collated_item[key] = torch.tensor([item[key] for item in items], dtype=torch.long)
        elif key in ['resolution', 'bounds']:
            # Keep as list of tuples (consistent with multi-dataset mode)
            collated_item[key] = [item[key] for item in items]
        else:
            collated_item[key] = [item[key] for item in items]

    for key in targets[0].keys():
        if isinstance(targets[0][key], torch.Tensor):
            collated_target[key] = torch.stack([target[key] for target in targets])
        elif key in ['resolution', 'bounds']:
            # Keep as list of tuples
            collated_target[key] = [target[key] for target in targets]
        else:
            collated_target[key] = [target[key] for target in targets]

    return collated_item, collated_target

class MultiDatasetPedestrianDataModule(pl.LightningDataModule):
    """
    DataModule that handles multiple datasets with different resolutions
    """

    def __init__(
            self,
            dataset_configs: List[Dict],  # List of dataset configurations
            batch_size: int = 2,
            num_workers: int = 4,
            accumulate_grad_batches: int = 8,
            shuffle_datasets: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.accumulate_grad_batches = accumulate_grad_batches
        self.shuffle_datasets = shuffle_datasets

        # Parse dataset configs
        self.dataset_configs = [DatasetConfig(**cfg) for cfg in dataset_configs]
        self.multi_config = MultiDatasetConfig(self.dataset_configs)

        # Storage for datasets
        self.datasets_train = {}
        self.datasets_val = {}
        self.datasets_test = {}

        # Storage for combined datasets
        self.data_train = None
        self.data_val = None
        self.data_test = None

        print(f"\n{'=' * 60}")
        print("Multi-Dataset DataModule Initialized")
        print(f"Number of datasets: {len(self.dataset_configs)}")
        for config in self.dataset_configs:
            print(f"\n  Dataset: {config.name}")
            print(f"    Resolution (Y,Z,X): {config.resolution}")
            print(f"    Bounds: {config.bounds}")
            print(f"    Cameras: {config.num_cameras}")
            print(f"    Sequence offset: {config.sequence_id_offset}")
        print(f"{'=' * 60}\n")

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each split"""

        if stage == 'fit' or stage is None:
            for config in self.dataset_configs:
                base = self._create_base_dataset(config, 'train')
                dataset = PedestrianDataset(
                    base,
                    is_train=True,
                    resolution=config.resolution,
                    bounds=config.bounds,
                )
                self.datasets_train[config.name] = dataset

            # Create combined training dataset
            self.data_train = CombinedDataset(
                self.datasets_train,
                self.multi_config
            )

        if stage == 'fit' or stage == 'validate' or stage is None:
            for config in self.dataset_configs:
                base = self._create_base_dataset(config, 'val')
                dataset = PedestrianDataset(
                    base,
                    is_train=False,
                    resolution=config.resolution,
                    bounds=config.bounds,
                )
                self.datasets_val[config.name] = dataset

            self.data_val = CombinedDataset(
                self.datasets_val,
                self.multi_config
            )

        if stage == 'test':
            for config in self.dataset_configs:
                base = self._create_base_dataset(config, 'test')
                dataset = PedestrianDataset(
                    base,
                    is_train=False,
                    resolution=config.resolution,
                    bounds=config.bounds,
                )
                self.datasets_test[config.name] = dataset

            self.data_test = CombinedDataset(
                self.datasets_test,
                self.multi_config
            )

    def _create_base_dataset(self, config: DatasetConfig, split: str):
        """Create base dataset object"""
        dataset_type = self._detect_dataset_type(config.data_dir)

        if 'mmcows' in dataset_type.lower():
            from datasets.tracktacular_mmcows_dataset import TrackTacularMMCows
            return TrackTacularMMCows(
                config.data_dir,
                split=split,
                sequences_file=config.sequences_file or 'sequences.json',
                sequence_id_offset=config.sequence_id_offset
            )
        elif 'tracktacular' in dataset_type.lower():
            from datasets.tracktacular_multiseq_dataset import TrackTacularMultiSeq
            return TrackTacularMultiSeq(
                config.data_dir,
                split=split,
                sequences_file=config.sequences_file or 'sequences.json',
                sequence_id_offset=config.sequence_id_offset
            )
        elif 'wildtrack' in dataset_type.lower():
            from datasets.wildtrack_dataset import Wildtrack
            return Wildtrack(config.data_dir)
        elif 'multiviewx' in dataset_type.lower():
            from datasets.multiviewx_dataset import MultiviewX
            return MultiviewX(config.data_dir)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def _detect_dataset_type(self, data_dir: str) -> str:
        """Detect dataset type from directory name"""
        dir_name = os.path.basename(data_dir).lower()
        if 'mmcows' in dir_name or 'mmcow' in dir_name:
            return 'mmcows'
        elif 'tracktacular' in dir_name:
            return 'tracktacular'
        elif 'wildtrack' in dir_name:
            return 'wildtrack'
        elif 'multiviewx' in dir_name:
            return 'multiviewx'
        else:
            return 'unknown'

    def train_dataloader(self):
        """Create training dataloader with temporal-chunk-aware multi-dataset sampler"""
        from datasets.multi_dataset_sampler import TemporalChunkDebugSampler

        batch_sampler = TemporalChunkDebugSampler(  # ← Changed from MultiDatasetBatchSampler
            dataset_frame_indices=self.data_train.get_dataset_frame_indices(),
            batch_size=self.batch_size,
            accumulate_grad_batches=self.accumulate_grad_batches,
            shuffle_datasets=True,
            drop_last=False,
        )

        print(f"\n{'=' * 60}")
        print("Training DataLoader (Temporal-Chunk Mode)")
        print(f"  Total batches per epoch: {len(batch_sampler)}")
        print(f"  Chunk size: {batch_sampler.chunk_size} frames")
        print(f"  Chunks will be shuffled across datasets each epoch")
        print(f"{'=' * 60}\n")

        return DataLoader(
            self.data_train,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.data_train.collate_fn,
        )

    def val_dataloader(self):
        """Create validation dataloader"""
        batch_sampler = MultiDatasetBatchSampler(
            dataset_frame_indices=self.data_val.get_dataset_frame_indices(),
            batch_size=self.batch_size,
            accumulate_grad_batches=self.accumulate_grad_batches,
            shuffle_datasets=False,
            drop_last=False,
        )

        return DataLoader(
            self.data_val,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.data_val.collate_fn,
        )

    def test_dataloader(self):
        """Create test dataloader - process one sample at a time"""
        return DataLoader(
            self.data_test,
            batch_size=1,
            num_workers=1,
            collate_fn=self.data_test.collate_fn,
        )


class CombinedDataset(torch.utils.data.Dataset):
    """
    Combines multiple PedestrianDatasets with different resolutions
    """

    def __init__(
            self,
            datasets: Dict[str, 'PedestrianDataset'],
            multi_config: MultiDatasetConfig
    ):
        self.datasets = datasets
        self.multi_config = multi_config

        # Build global index mapping
        self.index_to_dataset = []  # [(dataset_name, local_index), ...]
        self.dataset_frame_indices = defaultdict(list)  # {dataset_name: [global_indices]}

        global_idx = 0
        for dataset_name, dataset in datasets.items():
            for local_idx in range(len(dataset)):
                self.index_to_dataset.append((dataset_name, local_idx))
                self.dataset_frame_indices[dataset_name].append(global_idx)
                global_idx += 1

        self.total_length = len(self.index_to_dataset)

        print(f"\nCombinedDataset created:")
        print(f"  Total frames: {self.total_length}")
        for name, indices in self.dataset_frame_indices.items():
            print(f"    {name}: {len(indices)} frames")

    def __len__(self):
        return self.total_length

    def get_dataset_frame_indices(self):
        """Return mapping of dataset names to global frame indices"""
        return dict(self.dataset_frame_indices)

    def __getitem__(self, index):
        """Get item by global index"""
        dataset_name, local_index = self.index_to_dataset[index]
        dataset = self.datasets[dataset_name]

        item, target = dataset[local_index]

        # Add dataset metadata
        config = self.multi_config.get_config_by_name(dataset_name)
        item['dataset_name'] = dataset_name
        item['resolution'] = tuple(config.resolution)
        item['bounds'] = tuple(config.bounds)
        item['dataset_id'] = list(self.datasets.keys()).index(dataset_name)

        # Offset sequence numbers to prevent collisions
        if 'sequence_num' in item:
            item['sequence_num'] = item['sequence_num'] + config.sequence_id_offset

        target['resolution'] = tuple(config.resolution)
        target['bounds'] = tuple(config.bounds)

        return item, target

    def collate_fn(self, batch):
        """
        Custom collate function that handles variable resolutions.
        Ensures all items in a batch are from the same dataset.
        """
        if len(batch) == 0:
            return None, None

        # Verify all items are from same dataset
        dataset_names = [item[0]['dataset_name'] for item in batch]
        if len(set(dataset_names)) > 1:
            raise ValueError(
                f"Batch contains mixed datasets: {set(dataset_names)}. "
                "This should not happen with MultiDatasetBatchSampler."
            )

        # Standard collation
        items = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # Stack tensors
        collated_item = {}
        collated_target = {}

        for key in items[0].keys():
            if isinstance(items[0][key], torch.Tensor):
                collated_item[key] = torch.stack([item[key] for item in items])
            # Convert frame and sequence_num to tensors
            elif key in ['frame', 'sequence_num']:
                values = [item[key] for item in items]
                collated_item[key] = torch.tensor(values, dtype=torch.long)
            else:
                collated_item[key] = [item[key] for item in items]

        for key in targets[0].keys():
            if isinstance(targets[0][key], torch.Tensor):
                collated_target[key] = torch.stack([target[key] for target in targets])
            else:
                collated_target[key] = [target[key] for target in targets]

        return collated_item, collated_target


class PedestrianDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "../data/MultiviewX",
            batch_size: int = 2,
            num_workers: int = 4,
            resolution=None,
            bounds=None,
            accumulate_grad_batches=8,
            sequences_file: str = 'sequences.json',
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.resolution = resolution

        # Add validation
        if not isinstance(resolution, (tuple, list)):
            raise ValueError(f"resolution must be a tuple or list, got {type(resolution)}")

        if len(resolution) != 3:
            raise ValueError(f"resolution must have 3 elements (Y, Z, X), got {len(resolution)}: {resolution}")

        self.Y, self.Z, self.X = self.resolution

        self.bounds = bounds
        self.accumulate_grad_batches = accumulate_grad_batches
        self.sequences_file = sequences_file
        self.dataset = os.path.basename(self.data_dir)
        self.data_predict = None
        self.data_test = None
        self.data_val = None
        self.data_train = None

    def setup(self, stage: Optional[str] = None):
        if 'wildtrack' in self.dataset.lower():
            base = Wildtrack(self.data_dir)

        elif 'multiviewx' in self.dataset.lower():
            base = MultiviewX(self.data_dir)

        elif 'mmcows' in self.dataset.lower():
            # Multi-sequence mmCows dataset
            sequences_path = os.path.join(self.data_dir, self.sequences_file)
            if not os.path.exists(sequences_path):
                raise FileNotFoundError(
                    f"Sequences file not found: {sequences_path}\n"
                    f"For mmCows dataset, sequences file is required."
                )

            # Multi-sequence mode
            if stage == 'fit' or stage is None:
                base_train = TrackTacularMMCows(
                    self.data_dir,
                    split='train',
                    sequences_file=self.sequences_file
                )
                self.data_train = PedestrianDataset(
                    base_train,
                    is_train=True,
                    resolution=self.resolution,
                    bounds=self.bounds,
                )

            if stage == 'fit' or stage == 'validate' or stage is None:
                base_val = TrackTacularMMCows(
                    self.data_dir,
                    split='val',
                    sequences_file=self.sequences_file
                )
                self.data_val = PedestrianDataset(
                    base_val,
                    is_train=False,
                    resolution=self.resolution,
                    bounds=self.bounds,
                )

            if stage == 'test':
                base_test = TrackTacularMMCows(
                    self.data_dir,
                    split='test',
                    sequences_file=self.sequences_file
                )
                self.data_test = PedestrianDataset(
                    base_test,
                    is_train=False,
                    resolution=self.resolution,
                    bounds=self.bounds
                )

            if stage == 'predict':
                base_predict = TrackTacularMMCows(
                    self.data_dir,
                    split='test',
                    sequences_file=self.sequences_file
                )
                self.data_predict = PedestrianDataset(
                    base_predict,
                    is_train=False,
                    resolution=self.resolution,
                    bounds=self.bounds,
                )

            return  # Exit early - no need for single-sequence logic

        elif 'tracktacular' in self.dataset.lower():
            # Check if it's multi-sequence dataset by checking for sequences file
            sequences_path = os.path.join(self.data_dir, self.sequences_file)
            if os.path.exists(sequences_path):
                from datasets.tracktacular_multiseq_dataset import TrackTacularMultiSeq

                # Multi-sequence mode
                if stage == 'fit' or stage is None:
                    base_train = TrackTacularMultiSeq(
                        self.data_dir,
                        split='train',
                        sequences_file=self.sequences_file
                    )
                    self.data_train = PedestrianDataset(
                        base_train,
                        is_train=True,
                        resolution=self.resolution,
                        bounds=self.bounds,
                    )

                if stage == 'fit' or stage == 'validate' or stage is None:
                    base_val = TrackTacularMultiSeq(
                        self.data_dir,
                        split='val',
                        sequences_file=self.sequences_file
                    )
                    self.data_val = PedestrianDataset(
                        base_val,
                        is_train=False,
                        resolution=self.resolution,
                        bounds=self.bounds,
                    )

                if stage == 'test':
                    base_test = TrackTacularMultiSeq(
                        self.data_dir,
                        split='test',
                        sequences_file=self.sequences_file
                    )
                    self.data_test = PedestrianDataset(
                        base_test,
                        is_train=False,
                        resolution=self.resolution,
                        bounds=self.bounds
                    )

                if stage == 'predict':
                    base_predict = TrackTacularMultiSeq(
                        self.data_dir,
                        split='test',
                        sequences_file=self.sequences_file
                    )
                    self.data_predict = PedestrianDataset(
                        base_predict,
                        is_train=False,
                        resolution=self.resolution,
                        bounds=self.bounds,
                    )

                return  # Exit early for multi-sequence
            else:
                # Single-sequence mode - shouldn't be used for ablation study
                # but kept for backward compatibility
                from datasets.tracktacular_dataset import TrackTacular
                base = TrackTacular(self.data_dir)

        else:
            raise ValueError(f'Unknown dataset name {self.dataset}')

        # Single-sequence logic (original) - only executed for wildtrack, multiviewx, or single-sequence tracktacular
        if stage == 'fit' or stage is None:
            self.data_train = PedestrianDataset(
                base,
                is_train=True,
                resolution=self.resolution,
                bounds=self.bounds,
            )

        if stage == 'fit' or stage == 'validate' or stage is None:
            self.data_val = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )

        if stage == 'test':
            self.data_test = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds
            )

        if stage == 'predict':
            self.data_predict = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )

    def train_dataloader(self):
        sequence_frames = None
        if hasattr(self.data_train, 'sequence_frames'):
            sequence_frames = self.data_train.sequence_frames

        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=TemporalSampler(
                self.data_train,
                batch_size=self.batch_size,
                accumulate_grad_batches=self.accumulate_grad_batches,
                sequence_frames=sequence_frames
            ),
            pin_memory=True,
            collate_fn=single_dataset_collate_fn,
        )

    def val_dataloader(self):
        sequence_frames = None
        if hasattr(self.data_val, 'sequence_frames'):
            sequence_frames = self.data_val.sequence_frames

        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=TemporalSampler(
                self.data_val,
                batch_size=self.batch_size,
                accumulate_grad_batches=self.accumulate_grad_batches,
                sequence_frames=sequence_frames
            ),
            pin_memory=True,
            collate_fn=single_dataset_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,
            num_workers=1,
            collate_fn=single_dataset_collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )