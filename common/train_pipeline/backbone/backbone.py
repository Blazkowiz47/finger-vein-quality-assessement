"""
Factory for backbones.
"""


from common.train_pipeline.backbone.isotropic_backbone import IsotropicBackBone
from common.train_pipeline.config import BackboneConfig


def get_backbone(config: BackboneConfig):
    """
    Calls appropriate backbone build.
    """
    if config.backbone_type == "isotropic_backbone":
        return IsotropicBackBone(config.blocks)
