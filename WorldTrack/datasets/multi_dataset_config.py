"""
Multi-dataset configuration and utilities
"""
import json
from typing import Dict, List, Tuple


class DatasetConfig:
    """Configuration for a single dataset"""

    def __init__(
            self,
            name: str,
            data_dir: str,
            resolution: Tuple[int, int, int],  # Y, Z, X
            bounds: Tuple[float, float, float, float, float, float],
            num_cameras: int,
            depth: Tuple[int, float, float],
            sequences_file: str = None,
            z_sign: int = 1,
            sequence_id_offset: int = 0,
    ):
        self.name = name
        self.data_dir = data_dir
        self.resolution = resolution
        self.bounds = bounds
        self.num_cameras = num_cameras
        self.depth = depth
        self.sequences_file = sequences_file
        self.z_sign = z_sign
        self.sequence_id_offset = sequence_id_offset

    def to_dict(self):
        return {
            'name': self.name,
            'data_dir': self.data_dir,
            'resolution': self.resolution,
            'bounds': self.bounds,
            'num_cameras': self.num_cameras,
            'depth': self.depth,
            'sequences_file': self.sequences_file,
            'z_sign': self.z_sign,
            'sequence_id_offset': self.sequence_id_offset,
        }


class MultiDatasetConfig:
    """Configuration for multiple datasets"""

    def __init__(self, configs: List[DatasetConfig]):
        self.configs = configs
        self._assign_sequence_offsets()

    def _assign_sequence_offsets(self):
        """Assign sequence ID offsets to prevent collisions"""
        offset = 0
        for config in self.configs:
            config.sequence_id_offset = offset
            # Estimate max sequences (will be updated during dataset loading)
            offset += 1000  # Reserve 1000 IDs per dataset

    def get_config_by_name(self, name: str) -> DatasetConfig:
        """Get dataset config by name"""
        for config in self.configs:
            if config.name == name:
                return config
        raise ValueError(f"Dataset {name} not found in multi-dataset config")

    def get_all_resolutions(self) -> List[Tuple[int, int, int]]:
        """Get all unique resolutions across datasets"""
        return list(set(config.resolution for config in self.configs))

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load from YAML config file"""
        import yaml
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        configs = []
        for dataset_data in data['datasets']:
            config = DatasetConfig(**dataset_data)
            configs.append(config)

        return cls(configs)