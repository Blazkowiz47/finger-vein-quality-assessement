"""
 EER metric.
"""
from typing import Any, List, Optional

import matlab
import matlab.engine
import numpy as np
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
                if len(self.genuine.shape) == 0:
                    self.genuine = pred[self.genuine_label]
                    self.attack = pred[self.attack_indices]
                else:
                    self.genuine = torch.cat((self.genuine, pred[self.genuine_label]))
                    self.attack = torch.cat((self.attack, pred[self.attack_indices]))

        else:
            for pred, tar in zip(preds, target):
                genuine_index = tar == tar.max()
                if len(self.genuine.shape) == 0:
                    self.genuine = pred[genuine_index]
                    self.attack = pred[torch.logical_not(genuine_index)]
                else:
                    self.genuine = torch.cat((self.genuine, pred[genuine_index]))
                    self.attack = torch.cat(
                        (self.attack, pred[torch.logical_not(genuine_index)])
                    )

    def compute(self):
        genuine = self.genuine.detach().cpu().numpy()
        morphed = self.attack.detach().cpu().numpy()
        eer, far, ffr = self.eng.EER_DET_Spoof_Far(
            genuine, morphed, matlab.double(10000), nargout=3
        )
        far = np.array(far)
        ffr = np.array(ffr)
        one = np.argmin(np.abs(far - 1))
        pointone = np.argmin(np.abs(far - 0.1))
        pointzeroone = np.argmin(np.abs(far - 0.01))
        # _, _, _ = self.eng.Plot_ROC(genuine, morphed, matlab.double(10000), nargout=3)
        return (
            eer,
            100 - ffr[0][one],
            100 - ffr[0][pointone],
            100 - ffr[0][pointzeroone],
        )
