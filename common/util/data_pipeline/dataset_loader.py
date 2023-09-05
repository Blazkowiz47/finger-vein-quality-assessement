"""
    Contains base interface for datasets
"""


from abc import abstractmethod
import random
from typing import Any, Dict, Tuple, List
import numpy as np
from tqdm import tqdm
from common.util.enums import DatasetSplitType, EnvironmentType

from common.util.logger import logger
from common.util.models.dataset_models import DatasetObject


class DatasetLoaderBase:
    """Base class for all the datasets.
    This will act as an interface for loading a dataset."""

    def __init__(
        self,
        included_portion: float,
        isDatasetAlreadySplit: bool = False,
        train_portion: float = 0.7,
        validation_portion: float = 0.1,
        environment_type: EnvironmentType = EnvironmentType.NUMPY,
    ) -> None:
        self.environment_type: EnvironmentType = environment_type
        self.train_portion: float = train_portion
        self.validation_portion: float = validation_portion
        self.test_portion: float = 1 - train_portion - validation_portion
        self.included_portion: float = included_portion
        self.is_dataset_already_split: bool = isDatasetAlreadySplit
        self.files: Dict[DatasetSplitType, List[DatasetObject]] = {}

    @abstractmethod
    def get_directory(self) -> str:
        """Returns dataset's directory"""
        raise NotImplementedError()

    @abstractmethod
    def get_name(self) -> str:
        """Returns dataset's name"""
        raise NotImplementedError()

    def _convert_to_numpy_dataset(
        self, data_List: List[DatasetObject], split_type: DatasetSplitType
    ) -> List[np.ndarray]:
        pre_processed_data: List[DatasetObject] = []
        t_size = 0
        logger.info(f"Preprocessing {self.get_name()} dataset for {split_type.value} split.")
        for i, data in enumerate(tqdm(data_List)):
            pre_processed_data.append(self.pre_process(data))
            t_size = len(pre_processed_data[i])
        dataset = []
        for i in range(t_size):
            dataset.append(np.stack([data[i] for data in pre_processed_data]))
        return dataset

    def compile_sets(self) -> Dict[DatasetSplitType, List[np.ndarray]]:
        """Compiles train test and validation sets"""
        if self.is_dataset_already_split:
            self._populate_datasets_from_individual_pipelines()
        else:
            self._populate_datasets_from_all_files()

        result: Dict[DatasetSplitType, List[np.ndarray]] = {}
        for dataset_split_type, files in self.files.items():
            converted_dataset = self._convert_to_numpy_dataset(files, dataset_split_type)
            if converted_dataset:
                result[dataset_split_type] = converted_dataset
        return result

    def _sample(self, data: List[Any], portion: float):
        return random.sample(data, k=int(portion * len(data)))

    def _populate_datasets_from_individual_pipelines(self):
        self.files[DatasetSplitType.TRAIN] = self._sample(self.get_train_files(), self.included_portion)
        self.files[DatasetSplitType.TEST] = self._sample(self.get_test_files(), self.included_portion)
        self.files[DatasetSplitType.VALIDATION] = self._sample(self.get_validation_files(), self.included_portion)

    def _populate_datasets_from_all_files(self):
        """Pupulates train and test files from all_files"""
        self.all_files = self.get_files()
        included_files = self._sample(self.all_files, self.included_portion)
        self.files = self._split_train_test_validation(included_files)
        assert len(included_files) == (
            len(self.files[DatasetSplitType.TRAIN])
            + len(self.files[DatasetSplitType.TEST])
            + len(self.files[DatasetSplitType.VALIDATION])
        )

    def get_files(self) -> List[DatasetObject]:
        """Gets all the files at once"""
        return []

    def get_train_files(self) -> List[DatasetObject]:
        """Gets List of all the training files"""
        return []

    def get_test_files(self) -> List[DatasetObject]:
        """Gets List of all the test files"""
        return []

    def get_validation_files(self) -> List[DatasetObject]:
        """Gets List of all the validation files"""
        return []

    def _split_train_test_validation(self, data: List[DatasetObject]) -> Dict[DatasetSplitType, List[DatasetObject]]:
        """Splits data into train test and validation sets."""
        result: Dict[DatasetSplitType, List[DatasetObject]] = {}
        remaining_portions = self.test_portion + self.validation_portion
        remaining_portions = remaining_portions if remaining_portions else 1
        result[DatasetSplitType.TRAIN] = self._sample(data, self.train_portion)
        temp = [x for x in data if x not in result[DatasetSplitType.TRAIN]]
        result[DatasetSplitType.TEST] = self._sample(temp, (self.test_portion / remaining_portions))
        result[DatasetSplitType.VALIDATION] = [x for x in temp if x not in result[DatasetSplitType.TEST]]
        return result

    @abstractmethod
    def pre_process(self, data: DatasetObject) -> Tuple[np.ndarray, np.ndarray]:
        """
        Using the data, load the tensorflow image, along with labels/masks.
        To Remember:
        output of pre-process should be in following format:

        Example:
        def pre_process(self, data:DatasetObject):
            x = open(data.path)
            y = open(data.mask_path)
            x = _process(x)
            y = _process(y)
            return x,y
        """
        raise NotImplementedError()
