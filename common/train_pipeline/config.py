"""
Contains predefined configuration classes.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from common.gcn_lib.torch_vertex import GrapherConfig
from common.train_pipeline.backbone.attention_block import AttentionBlockConfig
from common.train_pipeline.backbone.ffn import FFNConfig
from common.train_pipeline.backbone.isotropic_backbone import (
    BackboneBlockConfig,
    IsotropicBlockConfig,
)
from common.train_pipeline.predictor.predictor import PredictorConfig
from common.train_pipeline.stem.stem import StemConfig

from common.util.enums import EnvironmentType


@dataclass
class BackboneConfig:
    """
    Backbone config.
    """

    backbone_type: str = "isotropic_backbone"
    blocks: List[BackboneBlockConfig] = None


class ModelConfig:
    """
    COntains Default training configurations.
    """

    def __init__(
        self,
        stem_config: Optional[StemConfig] = None,
        backbone_config: Optional[BackboneConfig] = None,
        predictor_config: Optional[PredictorConfig] = None,
    ):
        self.stem_config: Optional[StemConfig] = stem_config
        self.backbone_config: Optional[BackboneConfig] = backbone_config
        self.predictor_config: Optional[PredictorConfig] = predictor_config
