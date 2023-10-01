"""
Dataset generator for pytorch
"""
from typing import Any, List, Optional 
from torch.utils.data import DataLoader


def generate_dataset(
        data: List[Any],
        batch_size,
        shuffle: bool = True,
        num_workers: int = 2,
        total_files: Optional[int] = None,
        ) -> DataLoader:
    """
    Creates Dataset loader.
    """
    dataset_generator = DatasetGenerator(data, total_files= total_files)
    return DataLoader(
            dataset_generator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            )


class DatasetGenerator:
    """A generator which gives the dataset sequentially"""

    def __init__(
            self, data: List[Any], 
            total_files: Optional[int] = None,
            ) -> None:
        self.total_files = total_files
        self.data: List[Any] = data

    def __len__(self):
        if self.total_files:
            return self.total_files 
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
        # return (*[split[idx] for split in self.data],)
