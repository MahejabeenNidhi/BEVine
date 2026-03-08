# WorldTrack/datasets/sampler.py

from typing import Iterator, Sized
import torch
from torch.utils.data import Sampler


class TemporalSampler(Sampler[int]):
    """
    Original single-sequence temporal sampler.
    Unchanged from the baseline — kept for backward compatibility.
    """

    def __init__(self, data_source: Sized, batch_size: int = 2,
                 accumulate_grad_batches: int = 8) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        m = n - (n % (self.batch_size * self.accumulate_grad_batches))
        idx = torch.arange(m, dtype=torch.long).view(
            self.batch_size, self.accumulate_grad_batches, -1)
        idx = idx.transpose(0, 1).permute(
            *torch.arange(idx.ndim - 1, -1, -1)).flatten().tolist()
        idx = idx + list(range(m, n))
        yield from idx


class MultiSeqTemporalSampler(Sampler[int]):
    """
    Temporal sampler that respects sequence boundaries.

    Each sequence is sampled independently using the same interleaving
    logic as TemporalSampler so that:
      * frames within a batch come from the same sequence
      * the prev_bev temporal cache hits correctly across accumulation
        groups (consecutive "steps" differ by 1 in frame number)

    Sequences are optionally shuffled at each epoch (training) or kept
    in order (validation / test).
    """

    def __init__(self, data_source, batch_size: int = 2,
                 accumulate_grad_batches: int = 8,
                 shuffle_sequences: bool = True) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.shuffle_sequences = shuffle_sequences
        self.boundaries = data_source.get_sequence_boundaries()

    def __len__(self) -> int:
        return len(self.data_source)

    # ------------------------------------------------------------------
    def _sample_sequence(self, start: int, end: int) -> list:
        """Apply temporal interleaving within one sequence.

        Returns a list of *global* dataset indices.
        """
        seq_len = end - start
        effective_batch = self.batch_size * self.accumulate_grad_batches

        # Degenerate case: fewer frames than batch_size
        if seq_len < self.batch_size:
            return list(range(start, end))

        m = seq_len - (seq_len % effective_batch)

        if m == 0:
            # Sequence shorter than one full effective batch — use
            # simpler interleaving without the accumulate dimension.
            m2 = seq_len - (seq_len % self.batch_size)
            if m2 < self.batch_size:
                return list(range(start, end))
            local_idx = torch.arange(m2, dtype=torch.long).view(
                self.batch_size, -1).t().flatten().tolist()
            local_idx = local_idx + list(range(m2, seq_len))
            return [start + i for i in local_idx]

        # Full temporal interleaving (same logic as TemporalSampler)
        local_idx = torch.arange(m, dtype=torch.long).view(
            self.batch_size, self.accumulate_grad_batches, -1)
        local_idx = local_idx.transpose(0, 1).permute(
            *torch.arange(local_idx.ndim - 1, -1, -1)).flatten().tolist()
        local_idx = local_idx + list(range(m, seq_len))

        return [start + i for i in local_idx]

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[int]:
        seq_order = list(range(len(self.boundaries)))

        if self.shuffle_sequences:
            perm = torch.randperm(len(seq_order))
            seq_order = [seq_order[i] for i in perm.tolist()]

        all_indices = []
        for seq_idx in seq_order:
            start, end = self.boundaries[seq_idx]
            all_indices.extend(self._sample_sequence(start, end))

        yield from all_indices