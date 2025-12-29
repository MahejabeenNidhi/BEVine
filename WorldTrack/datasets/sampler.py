from typing import Iterator, Sized, Optional
import torch
from torch.utils.data import Sampler


class TemporalSampler(Sampler[int]):
    def __init__(
            self,
            data_source: Sized,
            batch_size: int = 2,
            accumulate_grad_batches: int = 8,
            sequence_frames: Optional[dict] = None
    ) -> None:
        """
        Args:
            data_source: Dataset
            batch_size: Batch size
            accumulate_grad_batches: Gradient accumulation steps
            sequence_frames: Dict mapping sequence_id to list of frame indices
                           If None, assumes single sequence
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.sequence_frames = sequence_frames

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        if self.sequence_frames is None:
            # Original single-sequence logic
            n = len(self.data_source)
            m = n - (n % (self.batch_size * self.accumulate_grad_batches))
            idx = torch.arange(m, dtype=torch.long).view(
                self.batch_size, self.accumulate_grad_batches, -1
            )
            idx = idx.transpose(0, 1).permute(*torch.arange(idx.ndim - 1, -1, -1)).flatten().tolist()
            idx = idx + list(range(m, n))
            yield from idx
        else:
            # Multi-sequence logic: process each sequence separately
            all_indices = []

            for seq_id in sorted(self.sequence_frames.keys()):
                seq_frame_count = len(self.sequence_frames[seq_id])

                # Get the starting index for this sequence
                start_idx = sum(len(self.sequence_frames[sid])
                                for sid in sorted(self.sequence_frames.keys())
                                if sid < seq_id)

                # Create temporal indices within this sequence
                m = seq_frame_count - (seq_frame_count % (self.batch_size * self.accumulate_grad_batches))

                if m > 0:
                    seq_idx = torch.arange(m, dtype=torch.long).view(
                        self.batch_size, self.accumulate_grad_batches, -1
                    )
                    seq_idx = seq_idx.transpose(0, 1).permute(
                        *torch.arange(seq_idx.ndim - 1, -1, -1)
                    ).flatten()
                    # Add offset to get global indices
                    seq_idx = seq_idx + start_idx
                    all_indices.extend(seq_idx.tolist())

                # Add remaining frames
                all_indices.extend(range(start_idx + m, start_idx + seq_frame_count))

            yield from all_indices
