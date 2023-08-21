"""
    Contains base interface for datasets
"""


from abc import abstractmethod
import random
from typing import Any, Tuple
import numpy as np
from common.utll.enums import SetType

from common.utll.models.dataset_models import DatasetObject


class DatasetLoaderBase:
    """Base class for all the datasets.
    This will act as an interface for loading a dataset."""

    def __init__(
        self,
        included_portion: float,
        isDatasetAlreadySplit: bool = False,
        train_portion: float = 0.7,
        validation_portion: float = 0.1,
    ) -> None:
        self.train_portion: float = train_portion
        self.validation_portion: float = validation_portion
        self.test_portion: float = 1 - train_portion - validation_portion
        self.included_portion: float = included_portion
        self.is_dataset_already_split: bool = isDatasetAlreadySplit
        self.files: dict[SetType, list[DatasetObject]] = {}

    @abstractmethod
    def get_directory(self) -> str:
        """Returns dataset's directory"""
        raise NotImplementedError()

    @abstractmethod
    def get_name(self) -> str:
        """Returns dataset's name"""
        raise NotImplementedError()

    def _convert_to_numpy_dataset(self, data_list: list[DatasetObject]) -> list[np.ndarray]:
        pre_processed_data: list[DatasetObject] = []
        t_size = 0
        for i, data in enumerate(data_list):
            pre_processed_data.append(self.pre_process(data))
            t_size = len(pre_processed_data[i])
        dataset = []
        for i in range(t_size):
            dataset.append(np.stack([data[i] for data in pre_processed_data]))
        return dataset

    def compile_sets(self) -> dict[SetType, list[np.ndarray]]:
        """Compiles train test and validation sets"""
        if self.is_dataset_already_split:
            self._populate_datasets_from_individual_pipelines()
        else:
            self._populate_datasets_from_all_files()

        result: dict[SetType, list[np.ndarray]] = {}
        for dataset_type, files in self.files.items():
            converted_dataset = self._convert_to_numpy_dataset(files)
            if converted_dataset:
                result[dataset_type] = converted_dataset
        return result

    def _sample(self, data: list[Any], portion: float):
        return random.sample(data, k=int(portion * len(data)))

    def _populate_datasets_from_individual_pipelines(self):
        self.files[SetType.TRAIN] = self._sample(self.get_train_files(), self.included_portion)
        self.files[SetType.TEST] = self._sample(self.get_test_files(), self.included_portion)
        self.files[SetType.VALIDATION] = self._sample(self.get_validation_files(), self.included_portion)

    def _populate_datasets_from_all_files(self):
        """Pupulates train and test files from all_files"""
        self.all_files = self.get_files()
        included_files = self._sample(self.all_files, self.included_portion)
        self.files = self._split_train_test_validation(included_files)
        assert len(included_files) == (
            len(self.files[SetType.TRAIN]) + len(self.files[SetType.TEST]) + len(self.files[SetType.VALIDATION])
        )

    def get_files(self) -> list[DatasetObject]:
        """Gets all the files at once"""
        return []

    def get_train_files(self) -> list[DatasetObject]:
        """Gets list of all the training files"""
        return []

    def get_test_files(self) -> list[DatasetObject]:
        """Gets list of all the test files"""
        return []

    def get_validation_files(self) -> list[DatasetObject]:
        """Gets list of all the validation files"""
        return []

    def _split_train_test_validation(self, data: list[DatasetObject]) -> dict[SetType, list[DatasetObject]]:
        """Splits data into train test and validation sets."""
        result: dict[SetType, list[DatasetObject]] = {}
        remaining_portions = self.test_portion + self.validation_portion
        remaining_portions = remaining_portions if remaining_portions else 1
        result[SetType.TRAIN] = self._sample(data, self.train_portion)
        temp = [x for x in data if x not in result[SetType.TRAIN]]
        result[SetType.TEST] = self._sample(temp, (self.test_portion / remaining_portions))
        result[SetType.VALIDATION] = [x for x in temp if x not in result[SetType.TEST]]
        return result

    @abstractmethod
    def pre_process(self, data: DatasetObject) -> Tuple[np.ndarray, np.ndarray]:
        """Using the data, load the tensorflow image, along with labels/masks"""
        raise NotImplementedError()
