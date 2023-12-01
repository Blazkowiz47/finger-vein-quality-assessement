"""
Contains various predictors.
"""
from dataclasses import dataclass
import torch
from torch.nn.functional import adaptive_avg_pool2d
from torch.nn import BatchNorm2d, Conv2d, Dropout, Linear, Module, Sequential, Softmax

from common.gcn_lib.torch_nn import act_layer


@dataclass
class PredictorConfig:
    """
    Predictor config.
    """

    predictor_type: str = "conv"
    in_channels: int = 1024
    n_classes: int = 600
    act: str = "relu"
    bias: bool = True
    hidden_dims: int = 2048
    dropout: float = 0.0
    conv_out_channels: int = 1024 // 4


def get_predictor(config: PredictorConfig) -> Module:
    """
    Builds appropriate predictor.
    """
    if config.predictor_type == "conv":
        return ConvPredictor(config)

    if config.predictor_type == "linear":
        return LinPredictor(config)


class ConvPredictor(Module):
    """
    Fully Convolutional Predictor.
    """

    def __init__(self, config: PredictorConfig):
        super(ConvPredictor, self).__init__()
        self.predictor = Sequential(
            Conv2d(config.in_channels, config.hidden_dims, 1, bias=config.bias),
            BatchNorm2d(config.hidden_dims),
            act_layer(config.act),
            Dropout(config.dropout),
        )
        self.fc1 = Linear(config.hidden_dims, config.n_classes)

        self.softmax = Softmax(dim=1)
        self.model_init()

    def forward(self, inputs):
        """Forward pass."""
        inputs = adaptive_avg_pool2d(inputs, 1)
        inputs = self.predictor(inputs).squeeze(-1).squeeze(-1)
        inputs = self.fc1(inputs)
        inputs = self.softmax(inputs)
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


class LinPredictor(Module):
    """
    Predictor with linear layers.
    """

    def __init__(self, config: PredictorConfig) -> None:
        super(LinPredictor, self).__init__()

        self.lin1 = Linear(config.in_channels, config.n_classes)
        self.softmax = Softmax(dim=1)

    def forward(self, inputs):
        """
        Forward pass.
        """
        inputs = adaptive_avg_pool2d(inputs, 1)
        inputs = inputs.squeeze(-1).squeeze(-1)
        inputs = self.lin1(inputs)
        inputs = self.softmax(inputs)
        return inputs
