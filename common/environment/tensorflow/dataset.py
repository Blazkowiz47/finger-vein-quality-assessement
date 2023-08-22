import tensorflow as tf
import numpy as np


def generate_dataset(data: list[np.ndarray]) -> tf.data.Dataset:
    tensorflow_datasets = [tf.data.Dataset.from_tensor_slices(data_or_label) for data_or_label in data]
    return tf.data.Dataset.zip((*tensorflow_datasets,))
