"""
    Compiles all the datasets
"""
from itertools import chain
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple
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

    def __init__(self, datasets: List[DatasetLoaderBase],
                 environment: EnvironmentType = EnvironmentType.NUMPY,
                 ) -> None:
        self.environment = environment
        self.datasets = datasets
        self.total_files: Dict[DatasetSplitType, Optional[int]] = {}
        self.compiled_datasets: Dict[DatasetSplitType, List[DatasetLoaderBase]] = {}
        for dataset in datasets:
            dataset.compile_sets()


    def _concatenate_datasets(
        self, split_type: DatasetSplitType, datasets: List[DatasetLoaderBase]
    ) -> List[DatasetLoaderBase]:
        """Concatenates all the datasets"""
        logger.info("Concatenating %s set", split_type.value)
        files = None
        for dataset in datasets:
            if dataset.total_files.get(split_type):
                if not files:
                    files = 0
                files += dataset.total_files[split_type]
        self.total_files[split_type] = files
        compiled_dataset: List[Any] = chain( 
                    *[
                        dataset.compiled_sets[split_type] 
                        for dataset in datasets
                    ]
                )
        return compiled_dataset

    def _get_dataset_converter(self, dataset_type: EnvironmentType) -> Any:
        module = import_module(f"common.util.environment.{dataset_type.value}.dataset")
        return getattr(module, "generate_dataset", None)
    

    def get_split(
            self,
            split_type: DatasetSplitType,
            batch_size: int = 10,
            shuffle: bool = False,
            ) -> Any:
        concatenated_dataset = self._concatenate_datasets(split_type, self.datasets)
        dataset_converter = self._get_dataset_converter(self.environment)
        return dataset_converter(concatenated_dataset, batch_size=batch_size, shuffle=shuffle, total_files=self.total_files[split_type],)

