import torch
from torch.nn import BatchNorm2d, Conv2d, Module, Sequential
from common.gcn_lib.torch_nn import act_layer
from common.train_pipeline.dscnet.Code.DRIVE.S3_DSConv_pro import DSConv_pro


class DSCModule(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 3,
        stride=1,
        bias=True,
    ) -> None:
        super(DSCModule, self).__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.xdsc = DSConv_pro(in_channels, out_channels, morph=0)
        self.ydsc = DSConv_pro(in_channels, out_channels, morph=1)
        self.enc = Conv2d(
            out_channels * 3,
            out_channels,
            kernel,
            stride=stride,
            padding=1,
            bias=bias,
        )

    def forward(self, inputs):
        c = self.conv(inputs)
        x = self.xdsc(inputs)
        y = self.ydsc(inputs)
        return self.enc(torch.cat([c, x, y], dim=1))


class DSCStem(Module):
    """
    Stem.
    """

    def __init__(
        self,
        in_dim=1,
        total_layers: int = 2,
        out_dim=256,
        act="relu",
        bias=True,
        requires_grad=True,
    ):
        super(DSCStem, self).__init__()
        self.layers = []
        start_channels = max(out_dim // pow(2, total_layers - 1), in_dim)
        for layer_number in range(total_layers):
            self.layers.append(
                DSCModule(
                    in_dim,
                    start_channels,
                    3,
                    stride=2 if layer_number + 1 != total_layers else 1,
                    bias=bias,
                )
            )
            self.layers.append(BatchNorm2d(start_channels))
            if layer_number + 1 != total_layers:
                self.layers.append(act_layer(act))
            in_dim = start_channels
            start_channels = start_channels * 2

        self.stem = Sequential(*self.layers)
        for parameter in self.stem.parameters():
            parameter.requires_grad = requires_grad

    def forward(self, inputs):
        """
        Forward pass.
        """
        inputs = self.stem(inputs)
        return inputs
