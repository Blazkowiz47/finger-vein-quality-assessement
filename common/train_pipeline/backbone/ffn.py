from torch.nn import BatchNorm2d, Conv2d, Identity, Module, Sequential
from timm.models.layers import DropPath
from common.gcn_lib.torch_nn import act_layer


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
    Output of this sequential network is added to original input. (Acts as skip connection).
    So make input of the layer can be added to output of the sequential network.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act="relu", drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Sequential(
            Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = Sequential(
            Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)
