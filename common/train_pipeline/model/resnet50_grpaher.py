import torch
from torch.nn import Module, Sequential
from common.train_pipeline.backbone.isotropic_backbone import IsotropicBackBone

from common.train_pipeline.config import ModelConfiguration


class Resnet50Grapher(Module):
    def __init__(self, config: ModelConfiguration) -> None:
        super(Resnet50Grapher, self).__init__()
        self.stem = torch.load(config.stem_config["path"])
        self.backbone = IsotropicBackBone(
            config=config.backbone_config,
        )
