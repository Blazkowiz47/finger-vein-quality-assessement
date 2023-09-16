# pylint disable=E1101, W0221
"""
Contains Confusion matrix.
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
        self.add_state(
            "false_positive_rate", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state(
            "true_positive_rate", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state(
            "false_negative_rate", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state(
            "true_negative_rate", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: Tensor,
        target: Tensor,
    ):
        assert preds.shape == target.shape
        for batch in range(preds.shape[0]):
            for class_id in range(preds.shape[1]):
                if target[batch][class_id] == preds[batch][class_id]:
                    if target[batch][class_id]:
                        self.true_positive_rate += 1
                    else:
                        self.true_negative_rate += 1
                else:
                    if target[batch][class_id]:
                        self.false_negative_rate += 1
                    else:
                        self.false_positive_rate += 1

                self.total += 1

    def compute(self):
        result: Dict[str, Any] = {
            "true_positive_rate": self.true_positive_rate.item() / self.total.item(),
            "true_negative_rate": self.true_negative_rate.item() / self.total.item(),
            "false_negative_rate": self.false_negative_rate.item() / self.total.item(),
            "false_positive_rate": self.false_positive_rate.item() / self.total.item(),
            # "total": self.total.item(),
        }
        self.reset()
        return result
