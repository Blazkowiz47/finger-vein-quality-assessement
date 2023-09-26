"""
evaluates everything
"""
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch import optim
from torch.nn import Module
from tqdm import tqdm
from timm.loss import SoftTargetCrossEntropy
from torchmetrics import Metric
from torchmetrics.classification import Accuracy
import wandb

# from common.data_pipeline.mmcbnu.dataset import DatasetLoader as mmcbnu
from common.train_pipeline.config import ModelConfig

# from common.data_pipeline.fvusm.dataset import DatasetLoader as fvusm
from common.data_pipeline.dataset import get_dataset

from common.train_pipeline.model.model import get_model
from common.util.logger import logger
from common.util.data_pipeline.dataset_chainer import DatasetChainer
from common.util.enums import EnvironmentType

# To watch nvidia-smi continuously after every 2 seconds: watch -n 2 nvidia-smi


def get_loss() -> Module:
    """
    Gets a loss function.
    """
    return SoftTargetCrossEntropy()


def get_metrics(n_classes: int) -> list[Metric]:
    """
    Returns list of metrics.
    """
    return [
        Accuracy(
            task="multiclass",
            num_classes=n_classes,
        ),
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
    datasets: List[str],
    model: str,
    batch_size: int = 10,
    environment: EnvironmentType = EnvironmentType.PYTORCH,
    augment_times: int = 0,
    n_classes: int = 301,
    height: int = 60,
    width: int = 120,
):
    """
    Contains the training loop.
    """
    device = cuda_info()
    train_dataset, test_dataset, validation_dataset = DatasetChainer(
        datasets=[
            get_dataset(
                dataset,
                environment=environment,
                augment_times=augment_times,
                height=height,
                width=width,
            )
            for dataset in datasets
        ],
    ).get_dataset(
        batch_size=batch_size,
        dataset_type=environment,
    )

    model = torch.load(model).to(device)
    logger.info(model)
    loss_fn = get_loss().to(device)

    metrics = [metric.to(device) for metric in get_metrics(n_classes)]
    # Training loop
    _ = cuda_info()
    with torch.no_grad():
        datasets = ["train", "test", "validation"]
        for index, dataset in enumerate(
            [train_dataset, test_dataset, validation_dataset]
        ):
            all_loss = []
            results = []
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
                predicted = outputs.argmax(dim=1)
                labels = labels.argmax(dim=1)
                for metric in metrics:
                    metric.update(predicted, labels)
            for metric in metrics:
                results.append(
                    add_label(
                        {
                            "accuracy": metric.compute().item(),
                            "loss": np.mean(all_loss),
                        },
                        datasets[index],
                    )
                )
                metric.reset()
            log = {}
            for result in results:
                log = log | result
            for k, v in log.items():
                logger.info("%s: %s", k, v)
