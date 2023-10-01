"""
Dataset generator for pytorch
"""
from typing import List, Optional, Tuple
import numpy as np
from torch.utils.data import DataLoader


def generate_dataset(
        data: List[np.ndarray],
        batch_size,
        shuffle: bool = True,
        num_workers: int = 2,
        total_files: Optional[int] = None,
        ) -> DataLoader:
    """
    Creates Dataset loader.
    """
    dataset_generator = DatasetGenerator(data)
    return DataLoader(
            dataset_generator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            )


class DatasetGenerator:
    """A generator which gives the dataset sequentially"""

    def __init__(
            self, data: List[Tuple[np.ndarray, np.ndarray]], 
            total_files: Optional[int] = None,
            ) -> None:
        self.total_files = total_files
        self.data: List[Tuple[np.ndarray, np.ndarray]] = data

    def __len__(self):
        if self.total_files:
            return self.total_files 
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
        # return (*[split[idx] for split in self.data],)
