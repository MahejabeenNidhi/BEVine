"""
Temporal-chunk-aware sampler with BALANCED interleaving to prevent catastrophic forgetting.
"""
from typing import Iterator, List, Dict
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict


class MultiDatasetBatchSampler(Sampler[List[int]]):
    """
    Sampler that ensures BALANCED representation of all datasets in every epoch.

    Key improvements:
    - Round-robin interleaving of chunks across datasets
    - Prevents catastrophic forgetting by ensuring dataset diversity in each epoch
    - Maintains temporal coherence within chunks
    """

    def __init__(
        self,
        dataset_frame_indices: Dict[str, List[int]],
        batch_size: int = 2,
        accumulate_grad_batches: int = 8,
        shuffle_datasets: bool = True,
        drop_last: bool = False,
        min_chunk_multiplier: int = 4,
    ):
        self.dataset_frame_indices = dataset_frame_indices  # Needed by _build_chunks_per_dataset()
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.shuffle_datasets = shuffle_datasets
        self.drop_last = drop_last

        base_chunk_size = batch_size * accumulate_grad_batches  # 16 frames
        self.chunk_size = base_chunk_size * min_chunk_multiplier  # 64 frames

        # Build chunks PER DATASET (not flat list)
        self.chunks_per_dataset = self._build_chunks_per_dataset()

        # Compute total batches
        self.total_batches = sum(
            sum((len(chunk['indices']) + batch_size - 1) // batch_size for chunk in chunks)
            for chunks in self.chunks_per_dataset.values()
        )

        print(f"\n{'='*70}")
        print("MultiDatasetBatchSampler (Balanced Interleaving Mode)")
        print(f"{'='*70}")
        print(f"Batch size: {batch_size}")
        print(f"Accumulate grad batches: {accumulate_grad_batches}")
        print(f"Chunk size: {self.chunk_size} consecutive frames")
        print(f"Total batches per epoch: {self.total_batches}")
        print(f"\nChunks per dataset:")
        for dataset_name, chunks in self.chunks_per_dataset.items():
            n_frames = sum(len(chunk['indices']) for chunk in chunks)
            print(f"  {dataset_name}: {len(chunks)} chunks ({n_frames} frames)")
        print(f"{'='*70}\n")

    def _build_chunks_per_dataset(self) -> Dict[str, List[Dict]]:
        """
        Build temporal chunks separately for each dataset.
        Returns: {dataset_name: [chunk1, chunk2, ...]}
        """
        chunks_per_dataset = {}

        for dataset_name, indices in self.dataset_frame_indices.items():
            chunks = []
            n_frames = len(indices)

            # Create complete chunks
            n_complete_chunks = n_frames // self.chunk_size
            for chunk_idx in range(n_complete_chunks):
                start = chunk_idx * self.chunk_size
                end = start + self.chunk_size
                chunks.append({
                    'dataset': dataset_name,
                    'indices': indices[start:end],
                })

            # Handle remaining frames
            if not self.drop_last:
                remaining_start = n_complete_chunks * self.chunk_size
                if remaining_start < n_frames:
                    chunks.append({
                        'dataset': dataset_name,
                        'indices': indices[remaining_start:],
                    })

            chunks_per_dataset[dataset_name] = chunks

        return chunks_per_dataset

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield batches with BALANCED round-robin interleaving.

        Strategy:
        1. Shuffle chunks WITHIN each dataset (preserves temporal order within chunks)
        2. Interleave chunks across datasets in round-robin fashion
        3. Every epoch sees balanced representation of all datasets
        """
        # Shuffle chunks within each dataset separately
        shuffled_chunks_per_dataset = {}
        for dataset_name, chunks in self.chunks_per_dataset.items():
            shuffled = chunks.copy()
            if self.shuffle_datasets:
                np.random.shuffle(shuffled)
            shuffled_chunks_per_dataset[dataset_name] = shuffled

        # Round-robin interleaving across datasets
        dataset_names = sorted(shuffled_chunks_per_dataset.keys())  # Deterministic order
        chunk_iterators = {
            name: iter(chunks)
            for name, chunks in shuffled_chunks_per_dataset.items()
        }

        active_datasets = set(dataset_names)
        dataset_idx = 0

        while active_datasets:
            # Select next dataset in round-robin order
            current_dataset = dataset_names[dataset_idx % len(dataset_names)]

            if current_dataset in active_datasets:
                try:
                    # Get next chunk from this dataset
                    chunk = next(chunk_iterators[current_dataset])
                    chunk_indices = chunk['indices']

                    # Yield batches from this chunk (in temporal order)
                    for i in range(0, len(chunk_indices), self.batch_size):
                        batch = chunk_indices[i:i + self.batch_size]
                        yield batch

                except StopIteration:
                    # This dataset exhausted, remove from active set
                    active_datasets.remove(current_dataset)

            dataset_idx += 1


class TemporalChunkDebugSampler(MultiDatasetBatchSampler):
    """
    Debug version that logs chunk transitions.
    Use this to verify balanced interleaving is working.
    """

    def __iter__(self) -> Iterator[List[int]]:
        shuffled_chunks_per_dataset = {}
        for dataset_name, chunks in self.chunks_per_dataset.items():
            shuffled = chunks.copy()
            if self.shuffle_datasets:
                np.random.shuffle(shuffled)
            shuffled_chunks_per_dataset[dataset_name] = shuffled

        dataset_names = sorted(shuffled_chunks_per_dataset.keys())
        chunk_iterators = {
            name: iter(chunks)
            for name, chunks in shuffled_chunks_per_dataset.items()
        }

        active_datasets = set(dataset_names)
        dataset_idx = 0
        chunk_count = 0
        prev_dataset = None

        print(f"\n{'='*70}")
        print("EPOCH START - Chunk Order:")
        print(f"{'='*70}")

        while active_datasets:
            current_dataset = dataset_names[dataset_idx % len(dataset_names)]

            if current_dataset in active_datasets:
                try:
                    chunk = next(chunk_iterators[current_dataset])
                    chunk_indices = chunk['indices']

                    # Log dataset transitions
                    if current_dataset != prev_dataset:
                        print(f"[Chunk {chunk_count:3d}] → {current_dataset:20s} ({len(chunk_indices)} frames)")
                        prev_dataset = current_dataset
                    chunk_count += 1

                    # Yield batches
                    for i in range(0, len(chunk_indices), self.batch_size):
                        batch = chunk_indices[i:i + self.batch_size]
                        yield batch

                except StopIteration:
                    active_datasets.remove(current_dataset)

            dataset_idx += 1

        print(f"{'='*70}\n")