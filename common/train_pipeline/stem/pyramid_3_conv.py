"""
Pyramid architecture stem.
"""
from torch.nn import BatchNorm2d, Conv2d, Module, Sequential

from common.gcn_lib.torch_nn import act_layer


class Pyramid3ConvStem(Module):
    """
    Pyramid architecture stem with 3 conv layers.
    2 for downsampling and 1 for increasing channel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: str,
        bias: bool = True,
    ) -> None:
        super(Pyramid3ConvStem, self).__init__()
        self.stem = Sequential(
            Conv2d(in_channels, out_channels // 2, 3, 2, 1, bias=bias),
            BatchNorm2d(out_channels // 2),
            act_layer(act),
            Conv2d(out_channels // 2, out_channels, 3, 2, 1, bias=bias),
            BatchNorm2d(out_channels),
            act_layer(act),
            Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias),
            BatchNorm2d(out_channels),
        )

    def forward(self, inputs):
        """
        Forward pass.
        """
        return self.stem(inputs)
