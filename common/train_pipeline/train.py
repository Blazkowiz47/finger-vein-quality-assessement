"""
Trains everything
"""
import time
from typing import Any, Dict
import torch
from torch import optim
import torch.autograd.profiler as profiler
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torchmetrics import Metric
import wandb

# from common.data_pipeline.MMCBNU_6000.dataset import DatasetLoader as mmcbnu
from common.train_pipeline.config import ModelConfig

# from common.data_pipeline.FV_USM.dataset import DatasetLoader as fvusm
from common.data_pipeline.dataset import DatasetLoader as common_dataset
from common.train_pipeline.metric.accuracy import Metric as Accuracy
from common.train_pipeline.metric.confusion_matrix import Metric as ConfusionMatrix

from common.train_pipeline.model.model import get_model
from common.util.logger import logger
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
            # fvusm(included_portion=1, environment_type=environment),
            common_dataset(
                "datasets/layer3output",
                "Internal_301_DB_layer3output",
                is_dataset_already_split=True,
                from_numpy=True,
                augment_times=0,
            )
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
    return [
        Accuracy().to(device),
        # ConfusionMatrix().to(device),
    ]


def get_test_metrics(device: str = "cpu") -> list[Metric]:
    """
    Returns list of testing metrics.
    """
    return [
        Accuracy().to(device),
        # ConfusionMatrix().to(device),
    ]


def get_val_metrics(device: str = "cpu") -> list[Metric]:
    """
    Returns list of validation metrics.
    """
    return [
        Accuracy().to(device),
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

    device = torch.device(
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


def train(
    config: ModelConfig,
    batch_size: int = 10,
    epochs: int = 1,
    environment: EnvironmentType = EnvironmentType.PYTORCH,
    log_on_wandb: bool = False,
    validate_after_epochs: int = 5,
    learning_rate: float = 1e-4,
):
    """
    Contains the training loop.
    """
    device = cuda_info()
    train_dataset, validation_dataset, _ = get_dataset(environment, batch_size)
    model = get_model(config).to(device)
    logger.info(model)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    train_loss_fn = get_train_loss(device)
    validate_loss_fn = get_val_loss(device)

    train_metrics = get_train_metrics(device)
    # test_metrics = get_test_metrics(device)
    val_metrics = get_val_metrics(device)
    # Training loop
    best_accuracy: float = 0
    _ = cuda_info()
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in tqdm(train_dataset, desc=f"Epoch {epoch} Training: "):
            # start = time.time()
            # with profiler.profile(record_shapes=True) as prof:
            inputs = inputs.cuda().float()
            labels = labels.cuda().float()
            # end = time.time()
            # logger.info("Leaded data on cuda. %s", str(end - start))
            optimizer.zero_grad()
            # start = time.time()
            outputs = model(inputs)  # pylint: disable=E1102
            # end = time.time()
            # logger.info("Forward prop. %s", str(end - start))
            loss = train_loss_fn(outputs, labels)
            # start = time.time()
            loss.backward()
            # end = time.time()
            # logger.info("Backward prop. %s", str(end - start))
            optimizer.step()
            predicted = (outputs == outputs.max()).float()
            # start = time.time()
            for metric in train_metrics:
                metric.update(predicted, labels)
            # end = time.time()
            # logger.info("Metric. %s", str(end - start))

        model.eval()
        computed_metrics = [metric.compute() for metric in train_metrics]
        results = [add_label(metric, "train") for metric in computed_metrics]

        if epoch % validate_after_epochs == 0:
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
        if best_accuracy < computed_metrics[0]["correct"]:
            torch.save(
                model,
                f"models/checkpoints/{log_on_wandb}.pt",
            )
        log = {}
        for result in results:
            log = log | result
        for k, v in log.items():
            logger.info("%s: %s", k, v)
        if log_on_wandb:
            wandb.log(log)

    model.train()
