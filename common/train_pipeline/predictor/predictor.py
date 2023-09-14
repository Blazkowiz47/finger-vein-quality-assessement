from torch.nn.functional import adaptive_avg_pool2d
from torch.nn import BatchNorm2d, Conv2d, Dropout, Module, Sequential

from common.gcn_lib.torch_nn import act_layer


class ConvPredictor(Module):
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

    def forward(self, x):
        x = adaptive_avg_pool2d(x, 1)
        return self.predictor(x).squeeze(-1).squeeze(-1)


class LinPredictor(Module):
    def __init__(
        self,
        channels=256,
        hdim=128,
        n_classes=100 * 6,
    ) -> None:
        super(LinPredictor, self).__init__()

    def forward(self, x):
        return x
