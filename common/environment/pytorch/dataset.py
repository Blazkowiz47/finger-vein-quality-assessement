import numpy as np
from torch.utils.data import DataLoader


def generate_dataset(data: list[np.ndarray], batch_size: int = 10, shuffle: bool = True):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
