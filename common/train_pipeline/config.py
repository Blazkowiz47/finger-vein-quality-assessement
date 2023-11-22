"""
Contains predefined configuration classes.
"""

from abc import ABC
from dataclasses import dataclass
from typing import List, Optional
from common.train_pipeline.predictor.predictor import PredictorConfig
from common.train_pipeline.stem.stem import StemConfig


class BackboneBlockConfig(ABC):
    """
    Abstract base class for backbone blocks.
    """

    block_type: str


@dataclass
class BackboneConfig:
    """
    Backbone config.
    """

    backbone_type: str = "isotropic_backbone"
    blocks: List[BackboneBlockConfig] = None
    requires_grad: bool = True


@dataclass
class ModelConfig:
    """
    Contains Default training configurations.
    """

    arcvein: bool = False
    stem_config: Optional[StemConfig] = None
    backbone_config: Optional[BackboneConfig] = None
    predictor_config: Optional[PredictorConfig] = None

    height: int = 60
    width: int = 120
