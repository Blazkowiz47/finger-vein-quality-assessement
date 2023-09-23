"""
    Compiles all the datasets
"""
from itertools import chain
from importlib import import_module
from typing import Any, Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from common.util.data_pipeline.dataset_loader import DatasetLoaderBase
from common.util.enums import DatasetSplitType, EnvironmentType
from common.util.models.dataset_models import DatasetObject
from common.util.logger import logger


class DatasetChainer:
    """
    Compiles all the datasets.
    Generates a Tensorflow compatible dataset.
    """

    def __init__(self, datasets: List[DatasetLoaderBase]) -> None:
        self.datasets = datasets
        self.files: Dict[DatasetSplitType, List[DatasetObject]] = {
            set_type: [] for set_type in DatasetSplitType
        }
        self.compiled_datasets: Dict[DatasetSplitType, List[np.ndarray]] = {}

    def _compile_all_datasets(self) -> None:
        """Compiles all the datasets together"""
        compiled_datasets: Dict[DatasetSplitType, List[List[np.ndarray]]] = {
            set_type: [] for set_type in DatasetSplitType
        }
        for dataset in self.datasets:
            sets = dataset.compile_sets()
            for set_type, data in sets.items():
                compiled_datasets[set_type].append(data)
                self.files[set_type].extend(dataset.files[set_type])
        self.compiled_datasets = {
            set_type: self._concatenate_datasets(set_type, dataset)
            for set_type, dataset in compiled_datasets.items()
        }

    def _concatenate_datasets(
        self, set_type: DatasetSplitType, datasets: List[Tuple[np.ndarray]]
    ) -> List[np.ndarray]:
        """Concatenates all the datasets"""
        logger.info("Concatenating %s set", set_type.value)
        compiled_dataset: List[np.ndarray] = chain(*datasets)
        return [data for data in tqdm(compiled_dataset)]

    def _get_dataset_converter(self, dataset_type: EnvironmentType):
        module = import_module(f"common.util.environment.{dataset_type.value}.dataset")
        return getattr(module, "generate_dataset", None)

    def _get_splits(
        self, dataset_type: EnvironmentType, batch_size: int = 1, shuffle: bool = False
    ) -> Tuple[Any]:
        dataset_converter = self._get_dataset_converter(dataset_type)
        return (
            *[
                dataset_converter(
                    self.compiled_datasets[split],
                    batch_size,
                    shuffle if not i else None,
                )
                for i, split in enumerate(self.compiled_datasets)
            ],
        )

    def get_dataset(
        self,
        dataset_type: EnvironmentType = EnvironmentType.NUMPY,
        batch_size: int = 10,
        shuffle: bool = False,
    ) -> Any:
        self._compile_all_datasets()
        return self._get_splits(dataset_type, batch_size, shuffle)

    def get_files(self, split_type: DatasetSplitType):
        return self.files.get(split_type, [])
