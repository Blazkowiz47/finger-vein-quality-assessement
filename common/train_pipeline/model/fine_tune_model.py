"""
FIne tuning pre trained model.
"""
import torch
from torch.nn import Conv2d, Module, Parameter
from common.train_pipeline.backbone.backbone import get_backbone

from common.train_pipeline.config import ModelConfig
from common.train_pipeline.predictor.predictor import get_predictor
from common.train_pipeline.stem.stem import get_stem
from common.util.logger import logger


class FineTuneModel(Module):
    """
    Model Wrapper
    """

    def __init__(self, config: ModelConfig, pretrained_model_path: str) -> None:
        super(FineTuneModel, self).__init__()
        self.predictor = None

        if config.predictor_config:
            self.predictor = get_predictor(
                config.predictor_config,
            )
        self.pretrained_model = torch.load(pretrained_model_path)
        for parameter in self.pretrained_model.parameters():
            parameter.requires_grad = False
        self.model_init()

    def forward(self, inputs):
        """
        Forward pass.
        """
        inputs = self.pretrained_model.stem(inputs)
        inputs = self.pretrained_model.backbone(inputs)

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
