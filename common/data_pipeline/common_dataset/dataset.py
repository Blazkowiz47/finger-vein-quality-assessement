"""
    Default dataset loader.
"""
import os
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
from common.util.data_pipeline.dataset_loader import DatasetLoaderBase
from common.util.decorators import reflected
from common.util.enums import EnvironmentType
from common.util.models.dataset_models import DatasetObject
from common.util.logger import logger


@reflected
class DatasetLoader(DatasetLoaderBase):
    """
    Default dataset loader.
    """

    def __init__(
        self,
        directory: str,
        dataset_name: str,
        environment_type: EnvironmentType = EnvironmentType.PYTORCH,
        included_portion: float = 1.0,
        train_size: float = 0.7,
        validation_size: float = 0.1,
        is_dataset_already_split: bool = False,
        from_numpy: bool = True,
        augment_times: int = 8,
        height: int = 60,
        width: int = 120,
    ) -> None:
        self.directory: str = directory
        self.dataset_name: str = dataset_name
        self.from_numpy = from_numpy
        self.height = height
        self.width = width
        self.classes: List[str] = self.get_classes()
        super().__init__(
            environment_type=environment_type,
            included_portion=included_portion,
            train_portion=train_size,
            validation_portion=validation_size,
            is_dataset_already_split=is_dataset_already_split,
            augment_times=augment_times,
        )

    def get_directory(self) -> str:
        return self.directory

    def get_classes(self) -> List[str]:
        """
        Gets all the classes.
        """
        try:
            classes = os.listdir(self.get_directory() + "/train")
        except FileNotFoundError:
            classes = os.listdir(self.get_directory())
        classes.sort()
        return classes

    def get_name(self) -> str:
        return self.dataset_name

    def loop_through_dir(self, directory: str) -> List[DatasetObject]:
        """
        Loops through  a directory.
        """
        result: List[DatasetObject] = []
        for index, class_label in enumerate(self.classes):
            images = os.listdir(f"{directory}/{class_label}")
            for image in images:
                if (
                    not image.endswith("bmp")
                    and not image.endswith("png")
                    and not image.endswith("jpg")
                ):
                    continue
                result.append(
                    DatasetObject(
                        path=f"{directory}/{class_label}/{image}",
                        name=f"{class_label}/{image.split('.')[0]}",
                        label=index,
                    )
                )
        return result

    def get_train_files(self) -> List[DatasetObject]:
        try:
            return self.loop_through_dir(f"{self.get_directory()}/train")
        except FileNotFoundError:
            logger.error("Error reading train dataset.")
            return []

    def get_test_files(self) -> List[DatasetObject]:
        try:
            return self.loop_through_dir(f"{self.get_directory()}/test")
        except FileNotFoundError:
            logger.error("Error reading test dataset.")
            return []

    def get_validation_files(self) -> List[DatasetObject]:
        try:
            return self.loop_through_dir(f"{self.get_directory()}/validation")
        except FileNotFoundError:
            logger.error("Error reading validation dataset.")
            return []

    def get_files(self) -> List[DatasetObject]:
        try:
            return self.loop_through_dir(self.get_directory())
        except FileNotFoundError:
            logger.error("Error reading datasets.")
            return []

    def pre_process(self, data: DatasetObject) -> Tuple[np.ndarray, np.ndarray]:
        if self.from_numpy:
            image = np.load(data.path, allow_pickle=True)
        else:
            image = cv2.imread(data.path, cv2.IMREAD_GRAYSCALE)  # pylint: disable=E1101
            if image is None:
                logger.info(data.path)
            if image.shape[0] > image.shape[1]:
                image = image.transpose()  # pylint: disable=E1101
            image = cv2.resize(  # pylint: disable=E1101
                image, (self.width, self.height)
            )
            image = (image - image.min()) / ((image.max() - image.min()) or 1.0)
            image = image.astype("float")
            if EnvironmentType.PYTORCH == self.environment_type:
                image = np.expand_dims(image, axis=0)
            elif EnvironmentType.TENSORFLOW == self.environment_type:
                image = np.expand_dims(image, axis=-1)
            image = np.vstack([image, image, image])

        label = np.zeros((len(self.classes)))
        label[data.label] = 1  # One hot encoding
        # return image, label
        return image.astype(np.float32), label

    def augment(self, image: np.ndarray, label: np.ndarray) -> List[np.ndarray]:
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.25),
                A.VerticalFlip(p=0.25),
                A.RandomBrightnessContrast(p=0.2),
                A.InvertImg(p=0.05),
                A.PixelDropout(p=0.02),
            ],
        )
        result: List[np.ndarray] = []
        for _ in range(self.augment_times):
            transformed = transform(image=image)
            transformed_image = transformed["image"]
            result.append(transformed_image)
        return result
