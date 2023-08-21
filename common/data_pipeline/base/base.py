"""
    Contains base interface for datasets
"""


from abc import abstractmethod
import random
from typing import Tuple
import numpy as np
from common.utll.enums import SetType

from common.utll.models.dataset_models import DatasetObject


class DatasetLoaderBase:
    """Base class for all the datasets.
    This will act as an interface for loading a dataset."""

    def __init__(
        self,
        train_size: float = 0.7,
        validation_size: float = 0.1,
    ) -> None:
        self.train_portion: float = train_size
        self.validation_portion: float = validation_size
        self.test_portion: float = 1 - train_size - validation_size
        self.all_files: list[DatasetObject] = self.get_files()
        self.files: dict[SetType, list[DatasetObject]] = {}

    @abstractmethod
    def get_directory(self) -> str:
        """Returns dataset's directory"""
        raise NotImplementedError()

    @abstractmethod
    def get_name(self) -> str:
        """Returns dataset's name"""
        raise NotImplementedError()

    def _convert_to_tf_dataset(self, data_list: list[DatasetObject]) -> list[np.ndarray]:
        t_size = 0
        for i, data in enumerate(data_list):
            data_list[i] = self.pre_process(data)
            t_size = len(data_list[i])
        dataset = []
        for i in range(t_size):
            dataset.append(np.stack([data[i] for data in data_list]))
        return dataset

    def compile_sets(self) -> dict[SetType, list[np.ndarray]]:
        """Compiles train test and validation sets"""
        if not self.all_files:
            self.files[SetType.TRAIN] = self.get_train_files()
            self.files[SetType.TEST] = self.get_test_files()
            self.files[SetType.VALIDATION] = self.get_validation_files()
        else:
            self._populate_train_test_validation_files()
        return {dataset_type: self._convert_to_tf_dataset(files) for dataset_type, files in self.files.items()}

    def _populate_train_test_validation_files(self):
        """Pupulates train and test files from all_files"""
        self.files = self._split_train_test_validation(self.all_files)
        assert len(self.all_files) == (
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
        number_of_train_samples = int(len(data) * self.train_portion)
        remaining_portions = self.test_portion + self.validation_portion
        remaining_portions = remaining_portions if remaining_portions else 1
        number_of_test_samples = int((self.test_portion / remaining_portions) * (len(data) - number_of_train_samples))
        result[SetType.TRAIN] = random.sample(self.all_files, k=number_of_train_samples)
        temp = [x for x in self.all_files if x not in self.files[SetType.TRAIN]]
        result[SetType.TEST] = random.sample(temp, k=number_of_test_samples)
        result[SetType.VALIDATION] = [x for x in temp if x not in result[SetType.TEST]]
        return result

    @abstractmethod
    def pre_process(self, data: DatasetObject) -> Tuple[np.ndarray, np.ndarray]:
        """Using the data, load the tensorflow image, along with labels/masks"""
        raise NotImplementedError()
