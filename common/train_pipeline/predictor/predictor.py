"""
Contains various predictors.
"""
from torch.nn.functional import adaptive_avg_pool2d
from torch.nn import BatchNorm2d, Conv2d, Dropout, Linear, Module, Sequential

from common.gcn_lib.torch_nn import act_layer
from common.train_pipeline.config import ModelConfiguration


def get_predictor(config: ModelConfiguration) -> Module:
    """
    Builds appropriate predictor.
    """
    config = config.predictor_config
    if config["name"] == "conv_predictor":
        return ConvPredictor(
            channels=config["channels"],
            hidden_channels=config.get("hidden_channels", config["channels"]),
            n_classes=config["n_classes"],
            act=config["act"],
            dropout=config.get("dropout"),
        )

    if config["name"] == "linear_predictor":
        return LinPredictor(
            channels=config["channels"],
            hidden_channels=config.get("hidden_channels", config["channels"]),
            n_classes=config["n_classes"],
            bias=config.get("bias", True),
            act=config.get("act", "relu"),
        )


class ConvPredictor(Module):
    """
    Fully Convolutional Predictor.
    """

    def __init__(
        self,
        channels=256,
        hidden_channels=512,
        n_classes=100 * 6,
        act="relu",
        dropout=0,
    ):
        super(ConvPredictor, self).__init__()
        self.predictor = Sequential(
            Conv2d(channels, hidden_channels, 1, bias=True),
            BatchNorm2d(hidden_channels),
            act_layer(act),
            Dropout(dropout),
            Conv2d(hidden_channels, n_classes, 1, bias=True),
        )

    def forward(self, inputs):
        """Forward pass."""
        inputs = adaptive_avg_pool2d(inputs, 1)
        return self.predictor(inputs).squeeze(-1).squeeze(-1)


class LinPredictor(Module):
    """
    Predictor with linear layers.
    """

    def __init__(
        self,
        channels=1024,
        hidden_channels=2048,
        n_classes=100 * 6,
        act="relu",
        bias=True,
    ) -> None:
        super(LinPredictor, self).__init__()
        self.predictor = Sequential(
            Conv2d(channels, channels, 3, 1, bias),
            Linear(channels, hidden_channels),
            act_layer(act),
            Linear(hidden_channels, n_classes),
        )

    def forward(self, inputs):
        """
        Forward pass.
        """
        inputs = self.predictor(inputs)
        return inputs
