"""
    Default dataset loader.
"""
from common.data_pipeline.dnp.dataset import DatasetLoader as dnp
from common.data_pipeline.fvusm.dataset import DatasetLoader as fvusm
from common.data_pipeline.mmcbnu.dataset import DatasetLoader as mmcbnu
from common.data_pipeline.common_dataset.dataset import DatasetLoader as common_dataset
from common.data_pipeline.post_process.dataset import DatasetLoader as post_process
from common.data_pipeline.morph.dataset import DatasetLoader as morph
from common.util.data_pipeline.dataset_loader import DatasetLoaderBase
from common.util.enums import EnvironmentType


def get_dataset(
    dataset: str,
    environment: EnvironmentType = EnvironmentType.PYTORCH,
    augment_times: int = 2,
    height: int = 60,
    width: int = 120,
) -> DatasetLoaderBase:
    """
    Dataset Factory.
    """
    if dataset == "mmcbnu":
        return mmcbnu(
            environment_type=environment,
            height=height,
            width=width,
            augment_times=augment_times,
        )

    if dataset == "fvusm":
        return fvusm(
            environment_type=environment,
            height=height,
            width=width,
            augment_times=augment_times,
        )
    if dataset == "layer3output":
        return common_dataset(
            "datasets/layer3output",
            "Resnet_Layer_3_Output",
            environment_type=environment,
            is_dataset_already_split=True,
            augment_times=augment_times,
            height=height,
            width=width,
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
            height=height,
            width=width,
        )
    if dataset.startswith("post_process"):
        dataset_split = dataset.split("_")
        printer = dataset_split[2]
        process_type = dataset_split[3]
        allowed_printers = ["DNP", "Digital", "Canon"]
        allowed_process_types = ["After", "Before"]
        if printer not in allowed_printers:
            raise ValueError(f"Printer not in :{allowed_printers}")
        if process_type not in allowed_process_types:
            raise ValueError(f"Process type not in :{allowed_process_types}")
        return post_process(
            printer,
            process_type,
            augment_times=augment_times,
            height=height,
            width=width,
        )

    if dataset.startswith("morph"):
        dataset_split = dataset.split("_")
        printer = dataset_split[1]
        morph_type = dataset_split[2]
        allowed_printers = ["dnp"]
        allowed_morph_types = [
            "cvmi",
            "lma",
            "lma_ucbo",
            "mipgan_1",
            "mipgan_2",
            "mordiff",
            "morphpipe",
            "stylegan",
        ]
        return morph(
            printer,
            morph_type,
            augment_times=augment_times,
            height=height,
            width=width,
        )
    raise ValueError("Dataset Not Implemented")
