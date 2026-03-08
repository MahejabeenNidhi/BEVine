# WorldTrack/datasets/multiseq_pedestrian_dataset.py

import bisect
from torch.utils.data import Dataset


class MultiSeqPedestrianDataset(Dataset):
    """
    Concatenates multiple PedestrianDataset instances (one per sequence).
    Tracks sequence boundaries so the sampler can respect them.

    Each sub-dataset covers a single sequence directory whose frames
    are used in their entirety (no 90/10 split — that is controlled
    by the use_all_frames flag inside PedestrianDataset).
    """

    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.cumulative_sizes = []
        self.sequence_boundaries = []

        cumsum = 0
        for ds in datasets:
            start = cumsum
            cumsum += len(ds)
            self.sequence_boundaries.append((start, cumsum))
            self.cumulative_sizes.append(cumsum)

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    # ------------------------------------------------------------------
    # Helpers used by the multi-sequence sampler
    # ------------------------------------------------------------------
    def get_sequence_boundaries(self):
        """Return list of (start, end) index pairs, one per sequence."""
        return self.sequence_boundaries

    @property
    def is_multi_seq(self):
        return True
