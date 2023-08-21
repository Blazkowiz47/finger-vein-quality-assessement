"""
    Dataset loader for dataset: MMCBNU_6000
"""
import os
from typing import Tuple

import cv2
from PIL import Image
import numpy as np
from common.data_pipeline.base.base import DatasetLoaderBase
from common.utll.decorators import reflected
from common.utll.models.dataset_models import DatasetObject


@reflected
class DatasetLoader(DatasetLoaderBase):
    """
    Dataset loader for dataset: MMCBNU_6000
    """

    def __init__(
        self,
        included_portion: float = 1.0,
        train_size: float = 0.7,
        validation_size: float = 0.1,
    ) -> None:
        self.fingers = ["Fore", "Middle", "Ring"]
        self.hands = ["L", "R"]
        super().__init__(
            included_portion=included_portion, train_portion=train_size, validation_portion=validation_size
        )

    def get_directory(self) -> str:
        return "./datasets/MMCBNU_6000"

    def get_name(self) -> str:
        return "MMCBNU6000"

    def get_files(self) -> list[DatasetObject]:
        dirs = os.listdir(self.get_directory() + "/ROIs")
        result: list[DatasetObject] = []
        for sample_id in dirs:
            for hand in self.hands:
                for finger in self.fingers:
                    images = os.listdir(f"{self.get_directory()}/ROIs/{sample_id}/{hand}_{finger}")
                    for image in images:
                        result.append(
                            DatasetObject(
                                path=f"{self.get_directory()}/ROIs/{sample_id}/{hand}_{finger}/{image}",
                                name=f"{sample_id}/{hand}_{finger}/{image}",
                                metadata={
                                    "finger": finger.lower(),
                                    "hand": "left" if hand == "L" else "right",
                                    "is_augmented": False,
                                },
                            )
                        )
        return result

    def pre_process(self, data: DatasetObject) -> Tuple[np.ndarray, np.ndarray]:
        image = Image.open(data.path)
        image = np.asarray(image)
        image = cv2.resize(image, dsize=(10, 10))
        return (image, np.array([1]).reshape((1, 1)))
