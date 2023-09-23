"""
    Dataset loader for dataset: dnp 
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


@reflected
class DatasetLoader(DatasetLoaderBase):
    """
    Dataset loader for dataset: MMCBNU_6000
    """

    def __init__(
        self,
        environment_type: EnvironmentType = EnvironmentType.PYTORCH,
        morph_type: str = "LMA",
        augment_times: int = 0,
    ) -> None:
        self.morph_type = morph_type
        self.classes = ["Morph", "Bonafide"]
        super().__init__(
            environment_type=environment_type,
            included_portion=1,
            is_dataset_already_split=True,
            augment_times=augment_times,
        )

    def get_directory(self) -> str:
        return "datasets/DNP_printer_canon_scan"

    def get_name(self) -> str:
        return "DNP Morphed vs Bonafide"

    def _list_dataset(self, morph_type: str, session: str) -> List[DatasetObject]:
        result: List[DatasetObject] = []
        for class_id, class_name in enumerate(self.classes):
            directory: str = (
                self.get_directory() + f"/{class_name}/{morph_type}/{session}"
            )
            if class_id:
                directory += "/Face_Detect_FullIm"
            images = os.listdir(directory)
            for image in images:
                if image.endswith("jpg"):
                    result.append(
                        DatasetObject(
                            path=f"{directory}/{image}",
                            name=image.split(".")[0],
                            label=[int(0 == class_id), int(1 == class_id)],
                        )
                    )
        return result

    def get_train_files(self) -> List[DatasetObject]:
        return self._list_dataset(self.morph_type, "ICAO_P2")

    def get_test_files(self) -> List[DatasetObject]:
        return self._list_dataset(self.morph_type, "ICAO_P1")

    def get_validation_files(self) -> List[DatasetObject]:
        return []

    def augment(self, image: np.ndarray) -> np.ndarray:
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.25),
                A.VerticalFlip(p=0.25),
                A.RandomBrightnessContrast(p=0.2),
                A.InvertImg(p=0.05),
                A.PixelDropout(p=0.02),
            ],
        )
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        transformed_image = (transformed_image - transformed_image.min()) / (
            transformed_image.max() - transformed_image.min()
        )
        transformed_image = transformed_image.astype(np.float32)
        return transformed_image

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
        return image, np.array(data.label)
