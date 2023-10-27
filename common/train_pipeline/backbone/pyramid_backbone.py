"""
Pyramid backbone.
"""
from dataclasses import dataclass
from typing import List, Optional
from torch.nn import Conv2d, BatchNorm2d, Module, Sequential
from common.gcn_lib.torch_vertex import Grapher, GrapherConfig
from common.train_pipeline.backbone.attention_block import (
    AttentionBlock,
    AttentionBlockConfig,
)
from common.train_pipeline.backbone.ffn import FFN, FFNConfig

from common.train_pipeline.config import BackboneBlockConfig


@dataclass
class PyramidBlockConfig(BackboneBlockConfig):
    """
    Pyramid backbone block config.
    """

    in_channels: int
    out_channels: int
    hidden_dimensions_in_ratio: int
    grapher_config: Optional[GrapherConfig] = None
    attention_config: Optional[AttentionBlockConfig] = None
    ffn_config: Optional[FFNConfig] = None
    number_of_nearest_neighbours: int = 9
    number_of_grapher_ffn_units: int = 2
    shrink_image_conv: bool = True


class PyramidBackbone(Module):
    """
    Pyramid backbone.
    """

    def __init__(self, config: List[PyramidBlockConfig], requires_grad=True) -> None:
        super(PyramidBackbone, self).__init__()
        self.layers: List[Sequential] = []
        for i, block in enumerate(config):
            self.layers.append(self.build_block(block))
            if block.shrink_image_conv:
                self.layers.append(
                    Conv2d(block.in_channels, block.out_channels, 3, 2, 1)
                )
                self.layers.append(BatchNorm2d(block.out_channels))
        self.backbone = Sequential(*self.layers)
        for param in self.backbone.parameters():
            param.requires_grad = requires_grad

    def forward(self, inputs):
        """
        Forward pass.
        """

        return self.backbone(inputs)

    def build_block(self, config: PyramidBlockConfig) -> Sequential:
        """
        Builds a pyramid block.
        """
        layers = []
        for _ in range(config.number_of_grapher_ffn_units):
            if config.grapher_config:
                layers.append(Grapher(config.grapher_config))
            if config.attention_config:
                layers.append(AttentionBlock(config.attention_config))
            if config.ffn_config:
                layers.append(FFN(config.ffn_config))
        return Sequential(*layers)
