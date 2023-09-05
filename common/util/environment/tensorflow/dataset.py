from tensorflow.data import Dataset
import numpy as np


def generate_dataset(data: List[np.ndarray], batch_size: int = 10, shuffle: bool = True) -> Dataset:
    tensorflow_datasets = [Dataset.from_tensor_slices(data_or_label) for data_or_label in data]
    dataset = Dataset.zip((*tensorflow_datasets,))
    dataset.batch(batch_size)
    if shuffle:
        dataset.shuffle(buffer_size=batch_size)
    return dataset
