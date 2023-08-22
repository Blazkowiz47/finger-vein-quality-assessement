import tensorflow as tf
import numpy as np


def generate_dataset(data: list[np.ndarray], batch_size: int = 10, shuffle: bool = True) -> tf.data.Dataset:
    tensorflow_datasets = [tf.data.Dataset.from_tensor_slices(data_or_label) for data_or_label in data]
    dataset = tf.data.Dataset.zip((*tensorflow_datasets,))
    dataset.batch(batch_size)
    if shuffle:
        dataset.shuffle(buffer_size=batch_size)
    return dataset
