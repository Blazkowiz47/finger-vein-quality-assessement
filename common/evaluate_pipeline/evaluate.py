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
    datasets: Union[List[str], Any],
    model_path: str,
    config: ModelConfig,
    batch_size: int = 10,
    environment: EnvironmentType = EnvironmentType.PYTORCH,
    augment_times: int = 0,
    n_classes: int = 301,
    height: int = 60,
    width: int = 120,
    eng: Any = None,
) -> Dict[str, Any]:
    """
    Contains the training loop.
    """
    device = cuda_info()
    if isinstance(datasets[0], str):
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
    else:
        train_dataset, test_dataset, validation_dataset = datasets

    model = get_model(config)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    # logger.info(model)
    loss_fn = get_loss().to(device)

    metrics = [metric.to(device) for metric in get_metrics(n_classes)]
    # Training loop
    with torch.no_grad():
        all_results: Dict[str, Any] = {}
        dataset_names = ["train", "test", "validation"]
        for index, dataset in enumerate(
            [train_dataset, test_dataset, validation_dataset]
        ):
            all_loss = []
            results = []
            scores: List[List[float]] = []
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
                for label, output in zip(labels, outputs):
                    scores.append(
                        [
                            label[0].item(),
                            label[1].item(),
                            output[0].item(),
                            output[1].item(),
                        ]
                    )
                predicted = outputs.argmax(dim=1)
                labels = labels.argmax(dim=1)
                for metric in metrics:
                    metric.update(predicted, labels)
            accuracy = None
            for metric in metrics:
                accuracy = metric.compute().item()
                results.append(
                    add_label(
                        {
                            "accuracy": accuracy,
                            "loss": np.mean(all_loss),
                        },
                        dataset_names[index],
                    )
                )
                metric.reset()
            scores = np.array(scores)
            precision = MulticlassPrecision(num_classes=2)
            recall = MulticlassRecall(num_classes=2)
            precision = precision(
                torch.from_numpy(scores[:, 2:]), torch.from_numpy(scores[:, :2])
            )
            recall = recall(
                torch.from_numpy(scores[:, 2:]), torch.from_numpy(scores[:, :2])
            )
            log = {}
            for result in results:
                log = log | result
            for k, v in log.items():
                logger.info("%s: %s", k, v)
            logger.info("Precision: %s", precision.item())
            logger.info("Recall: %s", recall.item())
            data = scores
            genuine = data[data[:, 1] == 1.0][:, 3]
            morphed = data[data[:, 0] == 1.0][:, 3]
            genuine = matlab.double(genuine.tolist())
            morphed = matlab.double(morphed.tolist())
            eer = None
            if eng:
                eer, _, _ = eng.EER_DET_Spoof_Far(
                    genuine, morphed, matlab.double(10000), nargout=3
                )
                logger.info("EER: %s", eer)
            else:
                eng = matlab.engine.start_matlab()
                try:
                    script_dir = f"/home/ubuntu/finger-vein-quality-assessement/EER"
                    eng.addpath(script_dir)
                except:
                    logger.exception("Cannot initialise matlab engine")
                    savemat(
                        f"results/{model_path.split('/')[-1].split('.')[0]}_{dataset_names[index]}_{datasets[0]}.mat",
                        {"genuine": genuine, "morphed": morphed},
                    )
            all_results[dataset_names[index]] = {
                "accuracy": accuracy,
                "precision": precision.item(),
                "recall": recall.item(),
                "eer": eer,
            }
        return all_results
