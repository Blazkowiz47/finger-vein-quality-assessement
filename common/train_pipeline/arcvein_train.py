"""
Trains everything
"""
from typing import Any, Dict, List, Optional
from itertools import chain
import numpy as np
import torch
from torch import optim
from torch.nn import Module
from torch.optim import lr_scheduler as lr_scheduler
from tqdm import tqdm
from torchmetrics import Metric
from torchmetrics.classification import Accuracy
import wandb

# from common.data_pipeline.mmcbnu.dataset import DatasetLoader as mmcbnu
from common.train_pipeline.config import ModelConfig

# from common.data_pipeline.fvusm.dataset import DatasetLoader as fvusm
from common.data_pipeline.dataset import get_dataset
from common.train_pipeline.arcvein import ArcCosineLoss, ArcVein
from common.metrics.eer import EER
from common.util.logger import logger
from common.util.data_pipeline.dataset_chainer import DatasetChainer
from common.util.enums import EnvironmentType
from torchmetrics.classification import Accuracy, MulticlassPrecision, MulticlassRecall
import matlab
import matlab.engine

# To watch nvidia-smi continuously after every 2 seconds: watch -n 2 nvidia-smi


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


def cuda_info() -> str:
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
    pretrained_predictor_classes: Optional[int] = None,
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

    loss_fn = ArcCosineLoss(
        n_classes, fine_tune=True if pretrained_model_path else False
    )
    model = ArcVein()

    if continue_model:
        model.load_state_dict(torch.load(continue_model))
        loss_fn.load_state_dict(torch.load(continue_model.split(".")[0] + "_loss.pt"))

    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path))

    model.to(device)
    loss_fn.to(device)
    logger.info(model)
    logger.info("Total parameters: %s", sum(p.numel() for p in model.parameters()))
    logger.info(
        "Total trainable parameters: %s",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    optimizer = optim.AdamW(
        loss_fn.parameters()
        if pretrained_model_path
        else chain(model.parameters(), loss_fn.parameters()),
        lr=learning_rate,
        weight_decay=0.05,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)

    train_metrics = [metric.to(device) for metric in get_train_metrics(n_classes, eng)]
    # test_metrics = get_test_metrics(device)
    val_metrics = [metric.to(device) for metric in get_val_metrics(n_classes, eng)]
    # Training loop
    best_train_accuracy: float = 0
    best_test_accuracy: float = 0
    best_eer: float = float("inf")
    _ = cuda_info()

    for epoch in range(1, epochs + 1):
        if pretrained_model_path:
            model.eval()
        else:
            model.train()
        loss_fn.train()
        training_loss = []
        for inputs, labels in tqdm(train_dataset, desc=f"Epoch {epoch} Training: "):
            logger.debug(
                "Inputs shape: %s, label shape: %s", inputs.shape, labels.shape
            )
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
            if pretrained_model_path:
                with torch.no_grad():
                    outputs = model(inputs)  # pylint: disable=E1102
            else:
                outputs = model(inputs)  # pylint: disable=E1102
            # end = time.time()
            # logger.info("Forward prop. %s", str(end - start))
            loss, outputs = loss_fn(
                outputs, labels, epochs < 30
            )  # pylint: disable=E1102
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
        loss_fn.eval()
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

        best_one, best_pointone, best_pointzerone = None, None, None

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
                    loss, outputs = loss_fn(
                        outputs, labels, False
                    )  # pylint: disable=E1102
                    val_loss.append(loss.item())

                    val_metrics[1].update(outputs, labels)

                    predicted = outputs.argmax(dim=1)
                    labels = labels.argmax(dim=1)
                    val_metrics[0].update(predicted, labels)
                eer, one, pointone, pointzeroone = val_metrics[1].compute()
                results.append(
                    add_label(
                        {
                            "accuracy": val_metrics[0].compute().item(),
                            "loss": np.mean(val_loss),
                            "eer": eer,
                            "tar1": one,
                            "tar0.1": pointone,
                            "tar0.01": pointzeroone,
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
                    torch.save(
                        loss_fn.state_dict(),
                        f"models/checkpoints/best_test_{log_on_wandb}_loss.pt",
                    )

                    best_test_accuracy = results[1]["test_accuracy"]
                if best_eer > results[1]["test_eer"]:
                    torch.save(
                        model.state_dict(),
                        f"models/checkpoints/best_eer_{log_on_wandb}.pt",
                    )
                    torch.save(
                        loss_fn.state_dict(),
                        f"models/checkpoints/best_eer_{log_on_wandb}_loss.pt",
                    )
                    best_eer = results[1]["test_eer"]
                    best_one = results[1]["test_tar1"]
                    best_pointone = results[1]["test_tar0.1"]
                    best_pointzerone = results[1]["test_tar0.01"]

        if best_train_accuracy < results[0]["train_accuracy"]:
            torch.save(
                model.state_dict(),
                f"models/checkpoints/best_train_{log_on_wandb}.pt",
            )
            torch.save(
                loss_fn.state_dict(),
                f"models/checkpoints/best_train_{log_on_wandb}_loss.pt",
            )
            best_train_accuracy = results[0]["train_accuracy"]

        log: Dict[str, Any] = {}
        for result in results:
            log = log | result
        for k, v in log.items():
            logger.info("%s: %s", k, v)
        if log_on_wandb:
            wandb.log(log)

        logger.info("Best EER:%s", best_eer)
        logger.info("Best TAR 1: %s", best_one)
        logger.info("Best TAR 0.1: %s", best_pointone)
        logger.info("Best TAR 0.01: %s", best_pointzerone)
        logger.info("Best test accuracy:%s", best_test_accuracy)
        logger.info("Best train accuracy:%s", best_train_accuracy)

    model.train()
