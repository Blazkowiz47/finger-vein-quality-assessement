import torch
from torch.nn import BatchNorm2d, Conv2d, Module, Sequential

from common.gcn_lib.torch_nn import act_layer


class Stem(Module):
    def __init__(self, img_shape=(60, 120), in_dim=1, out_dim=256, act="relu"):
        super(Stem, self).__init__()
        self.stem = Sequential(
            Conv2d(in_dim, out_dim // 8, 3, stride=2, padding=1),
            BatchNorm2d(out_dim // 8),
            act_layer(act),
            Conv2d(out_dim // 8, out_dim // 4, 3, stride=2, padding=1),
            BatchNorm2d(out_dim // 4),
            act_layer(act),
            Conv2d(out_dim // 4, out_dim // 2, 3, stride=2, padding=1),
            BatchNorm2d(out_dim // 2),
            act_layer(act),
            Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            BatchNorm2d(out_dim),
            act_layer(act),
            Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        return x
