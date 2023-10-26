"""
Stem factory.
"""
from dataclasses import dataclass
from typing import Optional
from torch.nn import Module
from common.train_pipeline.stem.conv_l5 import ConvStem
from common.train_pipeline.stem.pyramid_3_conv import Pyramid3ConvStem

from common.train_pipeline.stem.resnet50_l3 import Resnet50


@dataclass
class StemConfig:
    """
    Stem config.
    """

    stem_type: str = "pretrained_resnet50"
    pretrained: Optional[str] = "models/resnet50_pretrained.pt"
    resnet_layer: Optional[int] = 3
    in_channels: int = 3
    out_channels: int = 1024
    total_layers: int = 5
    act: str = "relu"
    requires_grad: bool = True
    bias: bool = True


def get_stem(config: StemConfig) -> Module:
    """
    Stem factory.
    """
    if config.stem_type == "pretrained_resnet50":
        if config.pretrained and config.resnet_layer:
            return Resnet50(
                path=config.pretrained,
                requires_grad=config.requires_grad,
                output_layer=config.resnet_layer,
            )
    if config.stem_type == "conv_stem":
        return ConvStem(
            in_dim=config.in_channels,
            out_dim=config.out_channels,
            act=config.act,
            total_layers=config.total_layers,
            bias=config.bias,
            requires_grad=config.requires_grad,
        )
    if config.stem_type == "pyramid_3_conv_layer":
        return Pyramid3ConvStem(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            act=config.act,
            bias=config.bias,
        )
    raise NotImplementedError(
        "No such stem has been implemented or theres some error in the config."
    )
