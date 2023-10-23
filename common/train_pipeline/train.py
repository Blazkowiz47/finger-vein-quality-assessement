"""
Trains everything
"""
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch import optim
from torch.nn import Module
from torch.optim import lr_scheduler as lr_scheduler
from tqdm import tqdm
from timm.loss import SoftTargetCrossEntropy
from torchmetrics import Metric
from torchmetrics.classification import Accuracy
import wandb

# from common.data_pipeline.mmcbnu.dataset import DatasetLoader as mmcbnu
from common.train_pipeline.config import ModelConfig

# from common.data_pipeline.fvusm.dataset import DatasetLoader as fvusm
from common.data_pipeline.dataset import get_dataset

from common.metrics.eer import EER
from common.train_pipeline.model.model import get_model
from common.util.logger import logger
from common.util.data_pipeline.dataset_chainer import DatasetChainer
from common.util.enums import EnvironmentType
from torchmetrics.classification import Accuracy, MulticlassPrecision, MulticlassRecall
import matlab
import matlab.engine

# To watch nvidia-smi continuously after every 2 seconds: watch -n 2 nvidia-smi


def get_train_loss() -> Module:
    """
    Gets a loss function for training.
    """
    return SoftTargetCrossEntropy()


def get_test_loss() -> Module:
    """
    Gets a loss function for training.
    """
    return SoftTargetCrossEntropy()


def get_val_loss() -> Module:
    """
    Gets a loss function for validation.
    """
    return SoftTargetCrossEntropy()


def get_train_metrics(n_classes: int, eng: Any) -> list[Metric]:
    """
    Returns list of training metrics.
    """
    return [
        Accuracy(
            task="multiclass",
            num_classes=n_classes,
        ),
        # EER(eng, genuine_class_label=1 if n_classes == 2 else None),
        # ConfusionMatrix().to(device),
    ]


def get_test_metrics(n_classes: int, eng: Any) -> list[Metric]:
    """
    Returns list of testing metrics.
    """
    return [
        Accuracy(
            task="multiclass",
            num_classes=n_classes,
        ),
        EER(eng, genuine_class_label=1 if n_classes == 2 else None),
        # ConfusionMatrix().to(device),
    ]


def get_val_metrics(n_classes: int, eng: Any) -> list[Metric]:
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


metric_names: List[str] = ["accuracy", "eer"]


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


def train(
    config: ModelConfig,
    dataset: str,
    batch_size: int = 10,
    epochs: int = 1,
    environment: EnvironmentType = EnvironmentType.PYTORCH,
    log_on_wandb: Optional[str] = None,
    validate_after_epochs: int = 5,
    learning_rate: float = 1e-3,
    continue_model: Optional[str] = None,
    augment_times: int = 0,
    n_classes: int = 301,
    height: int = 60,
    width: int = 120,
    pretrained_model_path: Optional[str] = None,
    eng: Any = None,
):
    """
    Contains the training loop.
    """

    try:
        eng = matlab.engine.start_matlab()
        script_dir = "/home/ubuntu/finger-vein-quality-assessement/EER"
        eng.addpath(script_dir)
    except Exception:
        logger.exception("Cannot initialise matlab engine")

    device = cuda_info()
    train_dataset, validation_dataset, _ = DatasetChainer(
        datasets=[
            get_dataset(
                dataset,
                environment=environment,
                augment_times=augment_times,
                height=height,
                width=width,
            ),
        ],
    ).get_dataset(
        batch_size=batch_size,
        dataset_type=environment,
    )

    if continue_model:
        model = get_model(config)
        model.load_state_dict(torch.load(continue_model))
        model = model.to(device)
    else:
        model = get_model(config, pretrained_model_path=pretrained_model_path).to(
            device
        )
    logger.info(model)
    logger.info("Total parameters: %s", sum(p.numel() for p in model.parameters()))
    logger.info(
        "Total trainable parameters: %s",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    train_loss_fn = get_train_loss().to(device)
    validate_loss_fn = get_val_loss().to(device)

    train_metrics = [metric.to(device) for metric in get_train_metrics(n_classes, eng)]
    # test_metrics = get_test_metrics(device)
    val_metrics = [metric.to(device) for metric in get_val_metrics(n_classes, eng)]
    # Training loop
    best_train_accuracy: float = 0
    best_test_accuracy: float = 0
    _ = cuda_info()
    scores: List[List[float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = []
        for inputs, labels in tqdm(train_dataset, desc=f"Epoch {epoch} Training: "):
            if inputs.shape[0] == 1:
                inputs = torch.cat((inputs, inputs), 0)  # pylint: disable=E1101
                labels = torch.cat((labels, labels), 0)  # pylint: disable=E1101
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
            loss = train_loss_fn(outputs, labels)  # pylint: disable=E1102
            # start = time.time()
            training_loss.append(loss.item())
            loss.backward()
            # end = time.time()
            # logger.info("Backward prop. %s", str(end - start))
            optimizer.step()
            # train_metrics[1].update(outputs, labels)
            # start = time.time()
            predicted = outputs.argmax(dim=1)
            labels = labels.argmax(dim=1)
            train_metrics[0].update(predicted, labels)
            # end = time.time()
            # logger.info("Metric. %s", str(end - start))

        scheduler.step()
        model.eval()
        results = []
        results.append(
            add_label(
                {
                    "accuracy": train_metrics[0].compute().item(),
                    # "eer": train_metrics[1].compute(),
                    "loss": np.mean(training_loss),
                },
                "train",
            )
        )
        for metric in train_metrics:
            metric.reset()

        if epoch % validate_after_epochs == 0:
            val_loss = []
            with torch.no_grad():
                for inputs, labels in tqdm(validation_dataset, desc="Validation:"):
                    if inputs.shape[0] == 1:
                        inputs = torch.cat((inputs, inputs), 0)  # pylint: disable=E1101
                        labels = torch.cat((labels, labels), 0)  # pylint: disable=E1101
                    inputs = inputs.to(device).float()
                    labels = labels.to(device).float()
                    outputs = model(inputs)  # pylint: disable=E1102
                    loss = validate_loss_fn(outputs, labels)  # pylint: disable=E1102
                    val_loss.append(loss.item())

                    val_metrics[1].update(outputs, labels)

                    predicted = outputs.argmax(dim=1)
                    labels = labels.argmax(dim=1)
                    val_metrics[0].update(predicted, labels)

                results.append(
                    add_label(
                        {
                            "accuracy": val_metrics[0].compute().item(),
                            "loss": np.mean(val_loss),
                            "eer": val_metrics[1].compute(),
                        },
                        "test",
                    )
                )
                for metric in val_metrics:
                    metric.reset()

                if best_test_accuracy < results[1]["test_accuracy"]:
                    torch.save(
                        model.state_dict(),
                        f"models/checkpoints/best_test_{log_on_wandb}.pt",
                    )
                    best_test_accuracy = results[1]["test_accuracy"]

        if best_train_accuracy < results[0]["train_accuracy"]:
            torch.save(
                model.state_dict(),
                f"models/checkpoints/best_train_{log_on_wandb}.pt",
            )
            best_train_accuracy = results[0]["train_accuracy"]

        log: Dict[str, Any] = {}
        for result in results:
            log = log | result
        for k, v in log.items():
            logger.info("%s: %s", k, v)
        if log_on_wandb:
            wandb.log(log)

    model.train()
