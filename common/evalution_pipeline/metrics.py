# pylint: disable=W0221, E1101
"""
Contains all the custom metrics.
"""
from typing import Any, Dict
import torch
from torch import Tensor
from torchmetrics import Metric


class Accuracy(Metric):
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
        for batch in range(preds.shape[0]):
            for class_id in range(preds.shape[1]):
                if target[batch][class_id] and target[batch][class_id] == preds[batch][class_id]:
                    self.correct += 1
                    break
        self.total += target.shape[0]

    def compute(self):
        result: Dict[str, Any] = {
            "correct": self.correct.item(),
            # "total": self.total.item(),
        }
        self.reset()
        return result
