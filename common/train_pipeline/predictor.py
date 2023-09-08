from torch.nn import BatchNorm2d, Conv2d, Dropout, Module, Sequential

from common.gcn_lib.torch_nn import act_layer


class Predictor(Module):
    def __init__(
        self,
        channels=256,
        hidden_channels=512,
        n_classes=100,
        act="relu",
        dropout=0,
    ):
        super(Predictor, self).__init__()
        self.predictor = Sequential(
            Conv2d(channels, hidden_channels, 1, bias=True),
            BatchNorm2d(hidden_channels),
            act_layer(act),
            Dropout(dropout),
            Conv2d(hidden_channels, n_classes, 1, bias=True),
        )

    def forward(self, x):
        return self.predictor(x).squeeze(-1).squeeze(-1)
