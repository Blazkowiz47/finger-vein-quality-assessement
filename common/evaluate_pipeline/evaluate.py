"""
evaluates everything
"""
from typing import Any, Dict, List, Tuple, Union
from common.train_pipeline.config import ModelConfig
from common.train_pipeline.model.model import get_model
import numpy as np
import torch
from torch.nn import Module
from tqdm import tqdm
from timm.loss import SoftTargetCrossEntropy
from torchmetrics import Metric
from torchmetrics.classification import Accuracy, MulticlassPrecision, MulticlassRecall
from scipy.io import savemat
import matlab
import matlab.engine

# from common.data_pipeline.mmcbnu.dataset import DatasetLoader as mmcbnu
# from common.data_pipeline.fvusm.dataset import DatasetLoader as fvusm
from common.data_pipeline.dataset import get_dataset

from common.metrics.eer import EER
from common.util.logger import logger
from common.util.data_pipeline.dataset_chainer import DatasetChainer
from common.util.enums import EnvironmentType
from train import get_config

# To watch nvidia-smi continuously after every 2 seconds: watch -n 2 nvidia-smi


def get_loss() -> Module:
    """
    Gets a loss function.
    """
    return SoftTargetCrossEntropy()


def get_metrics(n_classes: int, eng: Any) -> list[Metric]:
    """
    Returns list of validation metrics.
    """
    return [
        Accuracy(
            task="multiclass",
            num_classes=n_classes,
        ),
        EER(eng, genuine_class_label=1 if n_classes == 2 else None),
        # ConfusionMatrix().to(device),
    ]


def add_label(metric: Dict[str, Any], label: str = "") -> Dict[str, Any]:
    """
    Adds provided label as prefix to the keys in the metric dictionary.
    """
    return {f"{label}_{k}": v for k, v in metric.items()}


def cuda_info():
    """
    Prints cuda info.
    """

    device = torch.device(  # pylint: disable=E1101
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # pylint: disable=E1101
    logger.info("Using device: %s", device)

    # Additional Info when using cuda
    if device.type == "cuda":
        logger.info(torch.cuda.get_device_name(0))
        logger.info("Memory Usage:")
        logger.info(
            "Allocated: %s GB",
            round(torch.cuda.memory_allocated(0) / 1024**3, 1),
        )
        logger.info(
            "Cached: %s GB", round(torch.cuda.memory_reserved(0) / 1024**3, 1)
        )
    return device


def evaluate(
    datasets: Union[str, Any],
    model_path: str,
    config: ModelConfig,
    batch_size: int = 10,
    environment: EnvironmentType = EnvironmentType.PYTORCH,
    n_classes: int = 301,
    height: int = 60,
    width: int = 120,
) -> Dict[str, Any]:
    """
    Contains the training loop.
    """
    eng = matlab.engine.start_matlab()
    try:
        script_dir = "./EER"
        eng.addpath(script_dir)
    except Exception:
        logger.exception("Cannot initialise matlab engine")

    device = cuda_info()
    if isinstance(datasets, str):
        train_dataset, test_dataset, validation_dataset = DatasetChainer(
            datasets=[
                get_dataset(
                    datasets,
                    environment=environment,
                    augment_times=0,
                    height=height,
                    width=width,
                )
            ],
        ).get_dataset(
            batch_size=batch_size,
            dataset_type=environment,
        )
    else:
        train_dataset, test_dataset, validation_dataset = datasets

    model = get_model(config)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # logger.info(model)
    loss_fn = get_loss().to(device)
    metrics = [metric.to(device) for metric in get_metrics(n_classes, eng)]
    # Training loop
    with torch.no_grad():
        all_results: Dict[str, Any] = {}
        dataset_names = ["train", "test", "validation"]
        for index, dataset in enumerate(
            [train_dataset, test_dataset, validation_dataset]
        ):
            all_loss = []
            if not dataset:
                continue
            for inputs, labels in tqdm(dataset if dataset else [], desc="Train:"):
                if inputs.shape[0] == 1:
                    inputs = torch.cat((inputs, inputs), 0)  # pylint: disable=E1101
                    labels = torch.cat((labels, labels), 0)  # pylint: disable=E1101
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                outputs = model(inputs)  # pylint: disable=E1102
                loss = loss_fn(outputs, labels)  # pylint: disable=E1102
                all_loss.append(loss.item())
                metrics[1].update(outputs, labels)
                predicted = outputs.argmax(dim=1)
                labels = labels.argmax(dim=1)
                metrics[0].update(predicted, labels)

            accuracy = metrics[0].compute().item()
            eer, far, ffr = metrics[1].compute()
            logger.info("Evaluation results: %s", dataset_names[index])
            logger.info(
                "EER: %s\nFAR: %s\nFFR: %s",
                eer,
                np.array(far).shape,
                np.array(ffr).shape,
            )
            for metric in metrics:
                metric.reset()

            all_results[dataset_names[index]] = {
                "accuracy": accuracy,
                "eer": eer,
                "far": far,
                "ffr": ffr,
            }
        return all_results
