"""
    Default dataset loader.
"""
import os
from typing import List, Literal, Tuple, Union

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
        printer: Union[Literal["Canon"], Literal["DNP"], Literal["Digital"]],
        process_type: Union[Literal["After"], Literal["Before"]],
        environment_type: EnvironmentType = EnvironmentType.PYTORCH,
        augment_times: int = 8,
        height: int = 224,
        width: int = 224,
    ) -> None:
        self.height = height
        self.width = width
        self.printer = printer
        self.process_type = process_type
        self.classes = ["Mor", "Bon"]
        super().__init__(
            environment_type=environment_type,
            included_portion=1,
            augment_times=augment_times,
            is_dataset_already_split=True,
        )

    def get_directory(self) -> str:
        return f"/home/ubuntu/cluster/nbl-users/Shreyas-Sushrut-Raghu/PostProcess_Data/{self.printer}/{self.process_type}"

    def get_name(self) -> str:
        return self.printer + "_" + self.process_type

    def _list_dataset(self, split_type: str) -> List[DatasetObject]:
        result: List[DatasetObject] = []
        for class_id, class_name in enumerate(self.classes):
            directory: str = (
                self.get_directory() + "/" + class_name + "/" + split_type + "/Face"
            )
            images = os.listdir(directory)
            for image in images:
                if image.endswith("png") or image.endswith("png"):
                    result.append(
                        DatasetObject(
                            path=f"{directory}/{image}",
                            name=image.split(".")[0],
                            label=[int(0 == class_id), int(1 == class_id)],
                        )
                    )
        return result

    def get_train_files(self) -> List[DatasetObject]:
        return self._list_dataset("Train")

    def get_test_files(self) -> List[DatasetObject]:
        return self._list_dataset("Test")

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
        height, width = 224, 224
        image = cv2.imread(data.path)  # pylint: disable=E1101
        image = cv2.resize(image, (width, height))  # pylint: disable=E1101
        image = (image - image.min()) / (image.max() - image.min())
        if EnvironmentType.PYTORCH == self.environment_type:
            image = image.reshape((-1, 3))
            image = image.transpose()
            image = image.reshape((3, height, width))
        # return image, label
        image = image.astype(np.float32)
        return image, np.array(data.label).astype(np.float32)
