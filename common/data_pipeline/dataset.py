"""
    Compiles all the datasets
"""

import numpy as np
import tensorflow as tf
from common.data_pipeline.base.base import DatasetLoaderBase
from common.utll.enums import SetType
from common.utll.models.dataset_models import DatasetObject
from common.utll.logger import logger


class DatasetCompiler:
    """
    Compiles all the datasets.
    Generates a Tensorflow compatible dataset.
    """

    def __init__(self, datasets: list[DatasetLoaderBase]) -> None:
        self.datasets = datasets
        self.files: dict[SetType, list[DatasetObject]] = {set_type: [] for set_type in SetType}

    def compile_datasets(self) -> dict[SetType, tf.data.Dataset]:
        """Compiles all the datasets together"""
        compiled_datasets: dict[SetType, list[tf.data.Dataset]] = {set_type: [] for set_type in SetType}
        for dataset in self.datasets:
            sets = dataset.compile_sets()
            for set_type, data in sets.items():
                compiled_datasets[set_type].append(data)
                self.files[set_type].extend(dataset.files[set_type])

        result = {
            set_type: self.concatenate_datasets(set_type, dataset) for set_type, dataset in compiled_datasets.items()
        }
        return result

    def concatenate_datasets(self, set_type: SetType, datasets: list[np.ndarray]) -> tf.data.Dataset:
        """Concatenates all the datasets"""
        logger.info(f"Concatenating {set_type.value} set")
        t_size = 0
        for dataset in datasets:
            t_size = len(dataset)
            break
        compiled_dataset = []
        for i in range(t_size):
            compiled_dataset.append(np.concatenate([dataset[i] for dataset in datasets], axis=0))
        compiled_dataset = [tf.data.Dataset.from_tensor_slices(dataset) for dataset in compiled_dataset]
        return tf.data.Dataset.zip((*compiled_dataset,))
