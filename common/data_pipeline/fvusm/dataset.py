"""
    Dataset loader for dataset: FV_USM
"""
import os
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
from common.util.data_pipeline.dataset_loader import DatasetLoaderBase
from common.util.decorators import reflected
from common.util.enums import EnvironmentType
from common.util.logger import logger
from common.util.models.dataset_models import DatasetObject


@reflected
class DatasetLoader(DatasetLoaderBase):
    """
    Dataset loader for dataset: FV_USM
    """

    def __init__(
        self,
        environment_type: EnvironmentType = EnvironmentType.TENSORFLOW,
        included_portion: float = 1.0,
        train_size: float = 0.7,
        validation_size: float = 0.1,
        height: int = 60,
        width: int = 120,
        augment_times: int = 8,
        enhanced: bool = False,
    ) -> None:
        self.images = ["01", "02", "03", "04", "05", "06"]
        super().__init__(
            environment_type=environment_type,
            included_portion=included_portion,
            is_dataset_already_split=True,
        )
        self.enhanced = enhanced
        self.height = height
        self.width = width
        self.augment_times = augment_times
        self.n_classes = 492

    def get_directory(self) -> str:
        if self.enhanced:
            return "./datasets/enhanced_fvusm"
        return "./datasets/FV-USM"

    def get_name(self) -> str:
        if self.enhanced:
            return "Enhanced_FV_USM"
        return "FV_USM"

    def get_train_files(self) -> List[DatasetObject]:
        dirs = os.listdir(self.get_directory() + "/1st_session/extractedvein")
        dirs = [dir for dir in dirs if dir.startswith("vein")]
        dirs.sort()
        logger.info("Total classes: %s", len(dirs))
        result: List[DatasetObject] = []
        for class_id, sample_id in enumerate(dirs):
            for image in self.images:
                result.append(
                    DatasetObject(
                        path=f"{self.get_directory()}/1st_session/extractedvein/{sample_id}/{image}.jpg",
                        name=f"{sample_id[4:]}/{image}",
                        label=class_id,
                    )
                )

        return result

    def get_test_files(self) -> List[DatasetObject]:
        result: List[DatasetObject] = []
        dirs = os.listdir(self.get_directory() + "/2nd_session/extractedvein")
        dirs = [dir for dir in dirs if dir.startswith("vein")]
        dirs.sort()
        logger.info("Total classes: %s", len(dirs))
        for class_id, sample_id in enumerate(dirs):
            for image in self.images:
                result.append(
                    DatasetObject(
                        path=f"{self.get_directory()}/2nd_session/extractedvein/{sample_id}/{image}.jpg",
                        name=f"{sample_id[4:]}/{image}",
                        metadata={"finger": ""},
                        label=class_id,
                    )
                )
        return result

    def get_validation_files(self) -> List[DatasetObject]:
        return []

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
            transformed_image = (transformed_image - transformed_image.min()) / (
                transformed_image.max() - transformed_image.min()
            )
            transformed_image = transformed_image.astype(np.float32)
            result.append(transformed_image)
        return result

    def pre_process(self, data: DatasetObject) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(data.path, cv2.IMREAD_GRAYSCALE)  # pylint: disable=E1101
        image = cv2.resize(image, (self.height, self.width))
        image = np.asarray(image)
        image = image.transpose()
        image = (image - image.min()) / (image.max() - image.min())
        image = image.astype("float")
        if EnvironmentType.PYTORCH == self.environment_type:
            image = image.reshape((1, self.height, self.width))
        elif EnvironmentType.TENSORFLOW == self.environment_type:
            image = image.reshape((self.height, self.width, 1))
        label = np.zeros((self.n_classes,))
        label[data.label] = 1
        return np.vstack([image, image, image]).astype(np.float32), label.astype(
            np.float32
        )
