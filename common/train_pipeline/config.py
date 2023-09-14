"""
Contains predefined configuration classes.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import torch

from common.util.enums import EnvironmentType


class ModelConfiguration:
    """
    COntains Default training configurations.
    """

    stem_config: Dict[str, Any] = {
        "name": "resnet50",
        "train": False,
        "pretrained_model": "models/resnet50_pretrained.pt",
        "act": "gelu",
        "bias": True,
        "in_shape": (3, 60, 120),
        "in_dim": 3,
        "out_dim": 1024,
    }
    backbone_config: Dict[str, Any] = {
        "name": "isotropic_backbone",
        "train": True,
        "pretrained_model": None,
        "act": "gelu",
        "bias": True,
        "n_blocks": 12,
        "in_shape": (1024, 4, 8),
        "channels": 1024,
        "norm": "batch",
        "conv": "mr",
        "stochastic": False,
        "num_knn": 9,
        "use_dilation": False,
        "epsilon": 0.2,
        "drop_path": 0.0,
    }
    predictor_config: Dict[str, Any] = {
        "name": "conv",
        "train": True,
        "pretrained_model": None,
        "act": "gelu",
        "bias": True,
        "in_shape": (1024, 4, 8),
        "channels": 1024,
        "n_classes": 100 * 6,
        "hidden_channels": 2048,
    }

    batch_size: int = 10
    epochs: int = 1000
    checkpoint: int = 10

    train_loss: str = "cross_entropy_loss"
    test_loss: str = "cross_entropy_loss"
    validation_loss: str = "cross_entropy_loss"

    train_metrics: List[str] = ["accuracy"]
    test_metrics: List[str] = ["accuracy"]
    validation_metrics: List[str] = ["accuracy"]

    optimiser = None
    optimiser_lr: float = 1e-4

    number_of_augmented_images: int = 8
    input_shape: Tuple[int, int, int] = (3, 60, 120)
    shuffle: bool = True
    dataset: str = "MMCBNU_6000"

    environment: EnvironmentType = EnvironmentType.PYTORCH

    log_on_wandb: bool = True
    wandb_run_name: Optional[str] = None
    device: str = "cpu"

    def __init__(
        self,
        device: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        stem_config: Optional[Dict[str, Any]] = None,
        backbone_config: Optional[Dict[str, Any]] = None,
        predictor_config: Optional[Dict[str, Any]] = None,
    ):
        if wandb_run_name:
            self.wandb_run_name = wandb_run_name
        else:
            self.wandb_run_name = self.get_run_name()
        if stem_config:
            self.stem_config = self.stem_config | stem_config
        if backbone_config:
            self.backbone_config = self.backbone_config | backbone_config
        if predictor_config:
            self.predictor_config = self.predictor_config | predictor_config
        self.set_device(device)

    def set_device(self, device: Optional[str]):
        """
        Sets device used for training.
        """
        if device:
            self.device = device
        else:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )  # pylint: disable=E1101
            print("Using device:", device)

            # Additional Info when using cuda
            if device.type == "cuda":
                print(torch.cuda.get_device_name(0))
                print("Memory Usage:")
                print(
                    "Allocated:",
                    round(torch.cuda.memory_allocated(0) / 1024**3, 1),
                    "GB",
                )
                print(
                    "Cached:   ",
                    round(torch.cuda.memory_reserved(0) / 1024**3, 1),
                    "GB",
                )

    def get_run_name(self):
        "Compiles a run name automatically."
        current_time = datetime.now()
        stem = self.stem_config["name"]
        backbone = self.backbone_config["name"]
        predictor = self.predictor_config["name"]
        return f"{stem}_{backbone}_{predictor}_{current_time}"


default_gcn_backbone_config = {
    "name": "isotropic_backbone",
    "train": True,
    "pretrained_model": None,
    "act": "gelu",
    "bias": True,
    "n_blocks": 12,
    "in_shape": (1024, 4, 8),
    "channels": 1024,
    "norm": "batch",
    "conv": "mr",
    "stochastic": False,
    "num_knn": 9,
    "use_dilation": False,
    "epsilon": 0.2,
    "drop_path": 0.0,
}
