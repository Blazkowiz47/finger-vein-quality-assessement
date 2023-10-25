"""
FIne tuning pre trained model.
"""
import torch
from torch.nn import Conv2d, Module, Parameter
from common.train_pipeline.backbone.backbone import get_backbone

from common.train_pipeline.config import ModelConfig
from common.train_pipeline.model.custom_model import CustomModel
from common.train_pipeline.predictor.predictor import get_predictor
from common.train_pipeline.stem.stem import get_stem
from common.util.logger import logger


class FineTuneModel(Module):
    """
    Model Wrapper
    """

    def __init__(
        self,
        config: ModelConfig,
        pretrained_model_path: str,
        pretrained_predictor_classes: int,
    ) -> None:
        super(FineTuneModel, self).__init__()
        self.stem = None
        self.backbone = None
        self.predictor = None

        if config.predictor_config:
            self.predictor = get_predictor(
                config.predictor_config,
            )
        if config.predictor_config:
            config.predictor_config.n_classes = pretrained_predictor_classes
        pretrained_model = CustomModel(config)
        pretrained_model.load_state_dict(torch.load(pretrained_model_path))
        for parameter in pretrained_model.parameters():
            parameter.requires_grad = False
        if pretrained_model.stem:
            self.stem = pretrained_model.stem
        if pretrained_model.backbone:
            self.backbone = pretrained_model.backbone
        self.pos_embed = pretrained_model.pos_embed

        self.model_init()

    def forward(self, inputs):
        """
        Forward pass.
        """
        if self.stem:
            inputs = self.stem(inputs) + self.pos_embed
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
        for module in self.predictor.modules():
            if isinstance(module, Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
                module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.data.zero_()
                    module.bias.requires_grad = True
