"""
Stores model outputs.
"""
import os
import numpy as np
import torch

from torch.nn import Module
from tqdm import tqdm
from common.train_pipeline.stem.stem import StemConfig, get_stem
from common.util.data_pipeline.dataset_chainer import DatasetChainer

from common.util.enums import EnvironmentType

# from common.data_pipeline.MMCBNU_6000.dataset import DatasetLoader as mmcbnu
from common.data_pipeline.dataset import DatasetLoader as common_dataset

IMAGES_DONE_PER_CLASS = {}


def get_model(path: str) -> Module:
    """
    Gets pretrained model.
    """
    model = get_stem(StemConfig())
    return model


def model_output(model: Module, inputs) -> np.ndarray:
    """
    Gets pretrained model output.
    """
    output = model(inputs)
    output = output.cpu().numpy()
    return output


def save_outputs(
    images: np.ndarray,
    labels: np.ndarray,
):
    """
    Saves outputs.
    """
    assert images.shape[0] == labels.shape[0]
    for image, label in zip(images, labels):
        label = np.where(label == 1)[0][0]
        label += 1
        if IMAGES_DONE_PER_CLASS.get(label):
            np.save(f"{label}/{IMAGES_DONE_PER_CLASS[label] + 1}.npy", image)
            IMAGES_DONE_PER_CLASS[label] += 1
        else:
            np.save(f"{label}/{ + 1}.npy", image)
            IMAGES_DONE_PER_CLASS[label] = 1


def get_dataset(
    environment: EnvironmentType = EnvironmentType.PYTORCH,
    batch_size: int = 100,
):
    """
    Get's specific dataset within the provided environment.
    Change the datasets using config.
    """
    datasets = DatasetChainer(
        datasets=[
            # mmcbnu(
            #     included_portion=1,
            #     environment_type=environment,
            #     train_size=1.0,
            #     validation_size=0.0,
            #     isDatasetAlreadySplit=False,
            # ),
            # fvusm(included_portion=0, environment_type=environment),
            common_dataset(
                "datasets/internal_301_db",
                "Internal_301_DB",
                is_dataset_already_split=True,
                from_numpy=False,
                augment_times=3,
            )
        ]
    )
    return datasets.get_dataset(
        environment,
        shuffle=False,
        batch_size=batch_size,
    )


def initialise_dir(
    root_dir: str = "./datasets",
    output_dataset_name: str = "layer3output",
    n_classes: int = 600,
):
    """
    Initialises output dir.
    Creates dirs equal to the number of classes.
    """
    os.chdir(root_dir)
    os.mkdir(output_dataset_name)
    os.chdir(output_dataset_name)
    os.mkdir("test")
    os.mkdir("train")
    os.mkdir("validation")
    os.chdir("train")
    for label in range(1, n_classes + 1):
        os.mkdir(str(label))
    os.chdir("..")
    os.chdir("test")
    for label in range(1, n_classes + 1):
        os.mkdir(str(label))
    os.chdir("..")
    os.chdir("validation")
    for label in range(1, n_classes + 1):
        os.mkdir(str(label))
    os.chdir("..")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test, validation = get_dataset()
    MODEL = get_model("models/resnet50_pretrained.pt").to(device)
    initialise_dir(n_classes=301)
    os.chdir("train")
    all_labels = []
    for inputs, labels in tqdm(train):
        inputs = inputs.float().to(device)
        outputs = MODEL(inputs)
        all_labels.append(labels.cpu().numpy())
        save_outputs(outputs.cpu().numpy(), labels.cpu().numpy())

    os.chdir("..")
    os.chdir("test")
    IMAGES_DONE_PER_CLASS.clear()
    all_labels = []
    for inputs, labels in tqdm(test):
        inputs = inputs.float().to(device)
        outputs = MODEL(inputs)
        all_labels.append(labels.cpu().numpy())
        save_outputs(outputs.cpu().numpy(), labels.cpu().numpy())

    os.chdir("..")
    os.chdir("validation")
    IMAGES_DONE_PER_CLASS.clear()
    all_labels = []
    for inputs, labels in tqdm(validation):
        inputs = inputs.float().to(device)
        outputs = MODEL(inputs)
        all_labels.append(labels.cpu().numpy())
        save_outputs(outputs.cpu().numpy(), labels.cpu().numpy())
