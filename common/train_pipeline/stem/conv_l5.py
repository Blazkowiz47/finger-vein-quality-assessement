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
        out_dim=256,
        act="relu",
        bias=True,
    ):
        super(ConvStem, self).__init__()
        self.stem = Sequential(
            Conv2d(
                in_dim,
                out_dim // 8,
                3,
                stride=2,
                padding=1,
                bias=bias,
            ),
            BatchNorm2d(out_dim // 8),
            act_layer(act),
            Conv2d(
                out_dim // 8,
                out_dim // 4,
                3,
                stride=2,
                padding=1,
                bias=bias,
            ),
            BatchNorm2d(out_dim // 4),
            act_layer(act),
            Conv2d(
                out_dim // 4,
                out_dim // 2,
                3,
                stride=2,
                padding=1,
                bias=bias,
            ),
            BatchNorm2d(out_dim // 2),
            act_layer(act),
            Conv2d(
                out_dim // 2,
                out_dim,
                3,
                stride=2,
                padding=1,
                bias=bias,
            ),
            BatchNorm2d(out_dim),
            act_layer(act),
            Conv2d(
                out_dim,
                out_dim,
                3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            BatchNorm2d(out_dim),
        )

    def forward(self, inputs):
        """
        Forward pass.
        """
        inputs = self.stem(inputs)
        return inputs
