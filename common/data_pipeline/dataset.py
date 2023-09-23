"""
    Default dataset loader.
"""
from common.data_pipeline.dnp.dataset import DatasetLoader as dnp
from common.data_pipeline.fvusm.dataset import DatasetLoader as fvusm
from common.data_pipeline.mmcbnu.dataset import DatasetLoader as mmcbnu
from common.data_pipeline.common_dataset.dataset import DatasetLoader as common_dataset
from common.util.enums import EnvironmentType


def get_dataset(
    dataset: str,
    environment: EnvironmentType = EnvironmentType.PYTORCH,
    augment_times: int = 2,
):
    """
    Dataset Factory.
    """
    if dataset == "mmcbnu":
        return mmcbnu(
            environment_type=environment,
        )

    if dataset == "fvusm":
        return fvusm(
            environment_type=environment,
        )
    if dataset == "layer3output":
        return common_dataset(
            "datasets/layer3output",
            "Resnet_Layer_3_Output",
            environment_type=environment,
            is_dataset_already_split=True,
            augment_times=augment_times,
        )

    if dataset == "dnp_lma":
        return dnp(
            environment_type=environment,
            morph_type="LMA",
            augment_times=augment_times,
        )

    if dataset == "dnp_mipgan_1":
        return dnp(
            environment_type=environment,
            morph_type="MIPGAN_I",
            augment_times=augment_times,
        )
    if dataset == "dnp_mipgan_2":
        return dnp(
            environment_type=environment,
            morph_type="MIPGAN_II",
            augment_times=augment_times,
        )
    if dataset == "dnp_stylegan_iwbf":
        return dnp(
            environment_type=environment,
            morph_type="StyleGAN_IWBF",
            augment_times=augment_times,
        )

    if dataset == "internal_301_db":
        return common_dataset(
            f"datasets/{dataset}",
            "Internal_301",
            is_dataset_already_split=True,
            from_numpy=False,
            augment_times=augment_times,
        )
