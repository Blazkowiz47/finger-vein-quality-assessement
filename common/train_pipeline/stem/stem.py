import torch
from torch.nn import BatchNorm2d, Conv2d, Module, Sequential

from common.gcn_lib.torch_nn import act_layer


class Stem(Module):
    def __init__(
        self,
        img_shape=(60, 120),
        in_dim=1,
        out_dim=256,
        act="relu",
        pretrained=False,
    ):
        super(Stem, self).__init__()
        self.pretrained = pretrained
        if pretrained:
            self.stem = self.get_pretrained_stem()
        else:
            self.stem = self.get_stem(in_dim, out_dim, act)

    def get_pretrained_stem(self) -> Module:
        """
        Gets pretrained resnet50 stem.
        """
        self.stem = torch.load("models/resnet50_pretrained.pt")
        return self.stem

    def get_stem(
        self,
        in_dim: int,
        out_dim: int,
        act: str,
    ) -> Sequential:
        """
        Gets fresh stem.
        """
        return Sequential(
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
        if self.pretrained:
            x = self.stem.conv1(x)
            x = self.stem.bn1(x)
            x = self.stem.relu(x)
            x = self.stem.maxpool(x)
            x = self.stem.layer1(x)
            x = self.stem.layer2(x)
            x = self.stem.layer3(x)
            return x

        x = self.stem(x)
        return x
