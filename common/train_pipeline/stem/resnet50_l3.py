"""
Resnet50 pretrained stem.
"""
import torch
from torch.nn import Module


class Resnet50(Module):
    """
    Stem.
    """

    def __init__(
        self,
        path: str,
        requires_grad: bool = True,
        output_layer: int = 3,
    ):
        super(Resnet50, self).__init__()
        self.stem: Module = torch.load(path)
        for param in self.stem.parameters():
            param.requires_grad = requires_grad
        self.output_layer = output_layer

    def forward(self, inputs):
        """
        Forward pass.
        """
        inputs = self.stem.conv1(inputs)
        inputs = self.stem.bn1(inputs)
        inputs = self.stem.relu(inputs)
        inputs = self.stem.maxpool(inputs)
        if self.output_layer == 0:
            return inputs
        inputs = self.stem.layer1(inputs)
        if self.output_layer == 1:
            return inputs
        inputs = self.stem.layer2(inputs)
        if self.output_layer == 2:
            return inputs
        inputs = self.stem.layer3(inputs)
        if self.output_layer == 3:
            return inputs
        inputs = self.stem.layer4(inputs)
        return inputs
