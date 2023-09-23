"""
Contains various predictors.
"""
from dataclasses import dataclass
from torch.nn.functional import adaptive_avg_pool2d
from torch.nn import BatchNorm2d, Conv2d, Dropout, Linear, Module, Sequential, Softmax

from common.gcn_lib.torch_nn import act_layer
from common.util.logger import logger


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
    linear_dims: int = 1024 * 3
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

        self.conv1 = Conv2d(
            config.in_channels,
            config.in_channels * 2,
            3,
            1,
            bias=config.bias,
        )
        self.bn1 = BatchNorm2d(config.hidden_dims)
        self.act = act_layer(config.act)

        self.predictor = Sequential(
            Conv2d(
                config.in_channels * 2,
                config.hidden_dims,
                1,
                bias=config.bias,
            ),
            Dropout(config.dropout),
            Conv2d(
                config.hidden_dims,
                config.n_classes,
                1,
                bias=config.bias,
            ),
        )
        self.softmax = Softmax(dim=1)

    def forward(self, inputs):
        """Forward pass."""
        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs)
        inputs = self.act(inputs)
        inputs = adaptive_avg_pool2d(inputs, 1)
        inputs = self.predictor(inputs).squeeze(-1).squeeze(-1)
        inputs = self.softmax(inputs)
        return inputs


class LinPredictor(Module):
    """
    Predictor with linear layers.
    """

    def __init__(self, config: PredictorConfig) -> None:
        super(LinPredictor, self).__init__()
        self.conv1 = Conv2d(
            config.in_channels,
            config.in_channels * 2,
            3,
            1,
            bias=config.bias,
        )

        self.lin1 = Linear(config.linear_dims, config.n_classes)
        # self.act = act_layer(config.act)
        self.softmax = Softmax(dim=1)

    def forward(self, inputs):
        """
        Forward pass.
        """
        inputs = self.conv1(inputs)
        inputs = inputs.reshape((inputs.shape[0], -1))
        # inputs = inputs.squeeze(-1).squeeze(-1)
        # logger.debug("Ouput of after squeezing in linear predictor: %s", inputs.shape)
        inputs = self.lin1(inputs)
        # inputs = self.act(inputs)
        # logger.debug("Ouput of conv1 in linear predictor: %s", inputs.shape)
        # inputs = self.lin2(inputs)
        # logger.info(inputs.shape)
        inputs = self.softmax(inputs)
        return inputs
