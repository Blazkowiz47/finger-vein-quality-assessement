from typing import List
import numpy as np
from torch.utils.data import DataLoader


def generate_dataset(data: List[np.ndarray], batch_size, shuffle: bool = True) -> DataLoader:
    dataset_generator = DatasetGenerator(data)
    return DataLoader(dataset_generator, batch_size=batch_size, shuffle=shuffle)


class DatasetGenerator:
    """A generator which gives the dataset sequentially"""

    def __init__(self, data: List[np.ndarray]) -> None:
        self.data: List[np.ndarray] = data

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx: int):
        return (*[split[idx] for split in self.data],)
