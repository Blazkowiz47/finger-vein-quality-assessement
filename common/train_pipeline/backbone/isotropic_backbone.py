"""
Isotropic backbone.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from torch.nn import Module, Sequential
from common.gcn_lib.torch_vertex import Grapher, GrapherConfig
from common.train_pipeline.backbone.ffn import FFN, FFNConfig
from common.train_pipeline.backbone.attention_block import (
    AttentionBlock,
    AttentionBlockConfig,
)
from common.train_pipeline.config import BackboneBlockConfig


class SelfAttention(Enum):
    """
    Enum for self attention.
    """

    BEFORE = "before"
    AFTER = "after"
    BOTH = "both"


@dataclass
class IsotropicBlockConfig(BackboneBlockConfig):
    """
    Isotropic backbone config class.
    """

    block_type: str = "grapher_attention_ffn"
    grapher_config: Optional[GrapherConfig] = None
    ffn_config: Optional[FFNConfig] = None
    attention_config: Optional[AttentionBlockConfig] = None


class IsotropicBackBone(Module):
    """
    Isotropic Back bone.
    """

    def __init__(
        self,
        configs: List[IsotropicBlockConfig],
    ) -> None:
        super(IsotropicBackBone, self).__init__()

        layers: List[Sequential] = []
        for config in configs:
            layers.append(
                self.get_backbone_block(config),
            )
        self.backbone = Sequential(*layers)

    def forward(self, inputs):
        """
        Forward propogation.
        """
        # for i in range(self.n_blocks):
        inputs = self.backbone(inputs)
        return inputs

    def get_backbone_block(
        self,
        config: IsotropicBlockConfig,
    ) -> Sequential:
        """
        Gets backbone Block as per type.
        """
        if config.block_type == "grapher_attention_ffn":
            return Sequential(
                Grapher(config.grapher_config),
                AttentionBlock(config.attention_config),
                FFN(config.ffn_config),
            )

        if config.block_type == "attention_ffn":
            return Sequential(
                AttentionBlock(config.attention_config),
                FFN(config.ffn_config),
            )

        if config.block_type == "grapher_ffn":
            return Sequential(
                Grapher(config.grapher_config),
                FFN(config.ffn_config),
            )

        raise NotImplementedError("Invalid block type.")
