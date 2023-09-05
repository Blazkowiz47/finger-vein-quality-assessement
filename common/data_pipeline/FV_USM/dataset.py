"""
    Dataset loader for dataset: FV_USM
"""
import os
from typing import List, Tuple

import cv2
from PIL import Image
import numpy as np
from common.util.data_pipeline.dataset_loader import DatasetLoaderBase
from common.util.decorators import reflected
from common.util.enums import EnvironmentType
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
    ) -> None:
        self.images = ["01", "02", "03", "04", "05", "06"]
        super().__init__(
            environment_type=environment_type,
            included_portion=included_portion,
            train_portion=train_size,
            validation_portion=validation_size,
        )

    def get_directory(self) -> str:
        return "./datasets/FV-USM"

    def get_name(self) -> str:
        return "FV_USM"

    def get_files(self) -> List[DatasetObject]:
        dirs = os.listdir(self.get_directory() + "/1st_session/extractedvein")
        result: List[DatasetObject] = []
        for sample_id in dirs:
            if "vein" != sample_id[:4]:
                continue
            for image in self.images:
                result.append(
                    DatasetObject(
                        path=f"{self.get_directory()}/1st_session/extractedvein/{sample_id}/{image}.jpg",
                        name=f"{sample_id[4:]}/{image}",
                    )
                )

        dirs = os.listdir(self.get_directory() + "/2nd_session/extractedvein")
        for sample_id in dirs:
            if "vein" != sample_id[:4]:
                continue

            for image in self.images:
                result.append(
                    DatasetObject(
                        path=f"{self.get_directory()}/2nd_session/extractedvein/{sample_id}/{image}.jpg",
                        name=f"{sample_id[4:]}/{image}",
                        metadata={"finger": ""},
                    )
                )
        return result

    def pre_process(self, data: DatasetObject) -> Tuple[np.ndarray, np.ndarray]:
        image = Image.open(data.path)
        image = np.asarray(image)
        image = cv2.resize(image, dsize=(10, 10))
        image = (image - image.min()) / (image.max() - image.min())
        image = image.astype("float")
        if EnvironmentType.PYTORCH == self.environment_type:
            image = image.reshape((1, 10, 10))
        elif EnvironmentType.TENSORFLOW == self.environment_type:
            image = image.reshape((10, 10, 1))
        return image, np.array([1.0])
