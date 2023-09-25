"""
Resnet 50 grapher.
"""
import torch
from torch.nn import Conv2d, Module, Parameter
from common.train_pipeline.backbone.backbone import get_backbone

from common.train_pipeline.config import ModelConfig
from common.train_pipeline.predictor.predictor import get_predictor
from common.train_pipeline.stem.stem import get_stem
from common.util.logger import logger


class CustomModel(Module):
    """
    Model Wrapper
    """

    def __init__(self, config: ModelConfig) -> None:
        super(CustomModel, self).__init__()
        self.stem = None
        self.backbone = None
        self.predictor = None
        if config.stem_config:
            self.stem = get_stem(config.stem_config)
        if config.backbone_config:
            self.backbone = get_backbone(config.backbone_config)
        if config.predictor_config:
            self.predictor = get_predictor(
                config.predictor_config,
            )
        self.pos_embed = Parameter(
            torch.zeros(  # pylint: disable=E1101
                1, config.stem_config.out_channels, 224 // 4, 224 // 4
            )
        )
        self.model_init()

    def forward(self, inputs):
        """
        Forward pass.
        """
        if self.stem:
            inputs = self.stem(inputs)
        logger.debug("Stem output: %s", inputs.shape)
        if self.backbone:
            inputs = self.backbone(inputs)
        logger.debug("Backbone output: %s", inputs.shape)
        if self.predictor:
            inputs = self.predictor(inputs)
        logger.debug("Predictor output: %s", inputs.shape)
        return inputs

    def model_init(self):
        """
        Model init.
        """
        for module in self.modules():
            if isinstance(module, Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
                module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.data.zero_()
                    module.bias.requires_grad = True
