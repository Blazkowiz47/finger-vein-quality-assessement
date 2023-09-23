# pylint disable=E1101, W0221
"""
Contains accuracy metric.
"""
from typing import Any, Dict
import torch
from torch import Tensor
from torchmetrics import Metric as metric


class Metric(metric):
    """
    Measures total correct per run.
    """

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: Tensor,
        target: Tensor,
    ):
        assert preds.shape == target.shape
        self.correct += torch.sum((preds == target) * 1)
        self.total += target.shape[0]

    def compute(self):
        result: Dict[str, Any] = {
            "correct": self.correct.item() / self.total.item(),
            # "total": self.total.item(),
        }
        return result
