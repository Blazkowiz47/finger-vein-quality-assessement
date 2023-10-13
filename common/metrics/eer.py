"""
 EER metric.
"""
from typing import Any, List, Optional

import matlab
import matlab.engine
import torch
from torch import Tensor
from torchmetrics import Metric


class EER(Metric):
    def __init__(
        self,
        eng: Any,
        genuine_class_label: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.eng = eng
        self.add_state(
            "genuine",
            default=torch.tensor(0),
        )
        self.add_state(
            "attack",
            default=torch.tensor(0),
        )
        self.genuine_label = genuine_class_label
        self.attack_indices: Optional[List[int]] = None

    def update(self, preds: Tensor, target: Tensor):
        _, classes = target.shape

        if self.genuine_label:
            for pred, tar in zip(preds, target):
                if not self.attack_indices:
                    self.attack_indices = list(range(classes))
                    self.attack_indices.remove(self.genuine_label)

                self.genuine = torch.cat((self.genuine, pred[self.genuine_label]))
                self.attack = torch.cat((self.attack, pred[self.attack_indices]))

        else:
            for pred, tar in zip(preds, target):
                genuine = target == target.max()

                self.genuine = torch.cat((self.genuine, pred[genuine]))
                self.attack = torch.cat((self.attack, pred[torch.logical_not(genuine)]))

    def compute(self):
        genuine = self.genuine.detach().cpu().numpy()
        morphed = self.attack.detach().cpu().numpy()
        eer, _, _ = self.eng.EER_DET_Spoof_Far(
            genuine, morphed, matlab.double(10000), nargout=3
        )
        return eer
