"""
Numpy dataset generator.
"""
from typing import List
import numpy as np


def generate_dataset(
    data: List[np.ndarray], batch_size: int = 10, shuffle: bool = True
) -> List[np.ndarray]:
    return data
