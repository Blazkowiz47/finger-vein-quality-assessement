"""
Trains everything
"""
from typing import Any, Dict
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torchmetrics import Metric
import wandb

# from common.data_pipeline.MMCBNU_6000.dataset import DatasetLoader as mmcbnu
from common.train_pipeline.config import ModelConfig

from common.data_pipeline.FV_USM.dataset import DatasetLoader as fvusm
from common.train_pipeline.metric.accuracy import Metric as Accuracy

from common.train_pipeline.model.model import get_model
from common.util.data_pipeline.dataset_chainer import DatasetChainer
from common.util.enums import EnvironmentType

# To watch nvidia-smi continuously after every 2 seconds: watch -n 2 nvidia-smi


def get_dataset(
    environment: EnvironmentType = EnvironmentType.PYTORCH,
    batch_size: int = 10,
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
            #     train_size=0.85,
            #     validation_size=0.15,
            # ),
            fvusm(included_portion=1, environment_type=environment),
        ]
    )
    return datasets.get_dataset(environment, batch_size=batch_size)


def get_train_loss(device: str = "cpu"):
    """
    Gets a loss function for training.
    """
    if device == "cuda":
        return CrossEntropyLoss().cuda()
    return CrossEntropyLoss()


def get_test_loss(device: str = "cpu"):
    """
    Gets a loss function for training.
    """
    if device == "cuda":
        return CrossEntropyLoss().cuda()
    return CrossEntropyLoss()


def get_val_loss(device: str = "cpu"):
    """
    Gets a loss function for validation.
    """
    if device == "cuda":
        return CrossEntropyLoss().cuda()
    return CrossEntropyLoss()


def get_train_metrics(device: str = "cpu") -> list[Metric]:
    """
    Returns list of training metrics.
    """
    return [Accuracy().to(device)]


def get_test_metrics(device: str = "cpu") -> list[Metric]:
    """
    Returns list of testing metrics.
    """
    return [Accuracy().to(device)]


def get_val_metrics(device: str = "cpu") -> list[Metric]:
    """
    Returns list of validation metrics.
    """
    return [Accuracy().to(device)]


def add_label(metric: Dict[str, Any], label: str = "") -> Dict[str, Any]:
    """
    Adds provided label as prefix to the keys in the metric dictionary.
    """
    return {f"{label}_{k}": v for k, v in metric.items()}


def train(
    config: ModelConfig,
    batch_size: int = 10,
    epochs: int = 1,
    environment: EnvironmentType = EnvironmentType.PYTORCH,
    log_on_wandb: bool = False,
):
    """
    Contains the training loop.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # pylint: disable=E1101
    print("Using device:", device)

    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
    model = get_model(config).to(device)
    print(model)
    train_dataset, _, validation_dataset = get_dataset(environment, batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    train_loss_fn = get_train_loss(device)
    validate_loss_fn = get_val_loss(device)

    train_metrics = get_train_metrics(device)
    # test_metrics = get_test_metrics(device)
    val_metrics = get_val_metrics(device)
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in tqdm(train_dataset, desc=f"Epoch {epoch} Training: "):
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)  # pylint: disable=E1102
            loss = train_loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            predicted = (outputs == outputs.max()).float()
            for metric in train_metrics:
                metric.update(predicted, labels)
        model.eval()
        results = [add_label(metric.compute(), "train") for metric in train_metrics]

        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(validation_dataset, desc="Validation:"):
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                outputs = model(inputs)  # pylint: disable=E1102
                val_loss += validate_loss_fn(outputs, labels)
                predicted = (outputs == outputs.max()).float()
                for metric in val_metrics:
                    metric.update(predicted, labels)
        results.extend(
            [add_label(metric.compute(), "validation") for metric in val_metrics]
        )
        log = {}
        for result in results:
            log = log | result
        print(*[f"{k}: {v}" for k, v in log.items()])
        if log_on_wandb:
            wandb.log(log)

    model.train()
