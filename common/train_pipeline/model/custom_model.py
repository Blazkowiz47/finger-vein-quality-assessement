"""
Resnet 50 grapher.
"""
from torch.nn import Module
from common.train_pipeline.backbone.isotropic_backbone import IsotropicBackBone

from common.train_pipeline.config import ModelConfig
from common.train_pipeline.predictor.predictor import get_predictor
from common.train_pipeline.stem.stem import get_stem


class CustomModel(Module):
    """
    Resnet 50 Grapher.
    """

    def __init__(self, config: ModelConfig) -> None:
        super(CustomModel, self).__init__()
        self.stem = None
        self.backbone = None
        self.predictor = None
        if config.stem_config:
            self.stem = get_stem(config.stem_config)
        if config.backbone_config:
            self.backbone = IsotropicBackBone(
                config.backbone_config.blocks,
            )
        if config.predictor_config:
            self.predictor = get_predictor(
                config.predictor_config,
            )

    def forward(self, inputs):
        """
        Forward pass.
        """
        if self.stem:
            inputs = self.stem(inputs)
        if self.backbone:
            inputs = self.backbone(inputs)
        if self.predictor:
            inputs = self.predictor(inputs)
        return inputs
