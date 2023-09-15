"""
FFN module declaration.
"""
from dataclasses import dataclass
from typing import Optional
from torch.nn import BatchNorm2d, Conv2d, Identity, Module, Sequential
from timm.models.layers import DropPath
from common.gcn_lib.torch_nn import act_layer


@dataclass
class FFNConfig:
    """
    FFN config.
    """

    in_features: int
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    act: str = "relu"
    drop_path: float = 0.0
    bias: bool = True


class FFN(Module):
    """
    Feed Forward Network.
    Has following layers in a sequence:
    1. Conv2d ( in_features, hidden_features )
    2. BatchNorm2d ( hidden_features )
    3. Activation_Layer
    4. Conv2d ( hidden_features, out_features )
    5. BatchNorm2d ( out_features )
    6. Droppath if mentioned

    Note:
    Output of this sequential network is added to original input. (Acts as
    skip connection).
    So make input of the layer can be added to output of the sequential
    network.
    """

    def __init__(self, config: FFNConfig):
        super().__init__()
        out_features = config.out_features or config.in_features
        hidden_features = config.hidden_features or config.in_features
        self.fc1 = Sequential(
            Conv2d(
                config.in_features,
                hidden_features,
                1,
                stride=1,
                padding=0,
                bias=config.bias,
            ),
            BatchNorm2d(hidden_features),
        )
        self.act = act_layer(config.act)
        self.fc2 = Sequential(
            Conv2d(
                hidden_features,
                out_features,
                1,
                stride=1,
                padding=0,
                bias=config.bias,
            ),
            BatchNorm2d(out_features),
        )
        self.drop_path = (
            DropPath(config.drop_path) if config.drop_path > 0.0 else Identity()
        )

    def forward(self, inputs):
        """
        Forward pass.
        """
        shortcut = inputs
        inputs = self.fc1(inputs)
        inputs = self.act(inputs)
        inputs = self.fc2(inputs)
        inputs = self.drop_path(inputs) + shortcut
        return inputs  # .reshape(B, C, N, 1)
