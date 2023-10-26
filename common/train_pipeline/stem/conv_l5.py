"""
Conv stem. [VIG paper]
"""
from torch.nn import BatchNorm2d, Conv2d, Module, Sequential
from common.gcn_lib.torch_nn import act_layer


class ConvStem(Module):
    """
    Stem.
    """

    def __init__(
        self,
        in_dim=1,
        total_layers: int = 5,
        out_dim=256,
        act="relu",
        bias=True,
        requires_grad=True,
    ):
        super(ConvStem, self).__init__()
        self.layers = []
        start_channels = max(out_dim // pow(2, total_layers - 2), in_dim)
        for layer_number in range(total_layers):
            self.layers.append(
                Conv2d(
                    in_dim,
                    start_channels,
                    3,
                    stride=2 if layer_number + 1 != total_layers else 1,
                    padding=1,
                    bias=bias,
                )
            )
            self.layers.append(BatchNorm2d(start_channels))
            if layer_number + 1 != total_layers:
                self.layers.append(act_layer(act))
            in_dim = start_channels
            start_channels = (
                start_channels * 2
                if layer_number + 2 != total_layers
                else start_channels
            )

        self.stem = Sequential(*self.layers)
        for parameter in self.stem.parameters():
            parameter.requires_grad = requires_grad

    def forward(self, inputs):
        """
        Forward pass.
        """
        inputs = self.stem(inputs)
        return inputs
