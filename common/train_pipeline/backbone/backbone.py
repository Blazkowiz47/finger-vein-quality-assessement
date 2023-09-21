"""
Factory for backbones.
"""


from common.train_pipeline.backbone.isotropic_backbone import IsotropicBackBone
from common.train_pipeline.backbone.pyramid_backbone import PyramidBackbone
from common.train_pipeline.config import BackboneConfig


def get_backbone(config: BackboneConfig):
    """
    Calls appropriate backbone build.
    """
    if config.backbone_type == "isotropic_backbone":
        return IsotropicBackBone(config.blocks)
    if config.backbone_type == "pyramid_backbone":
        return PyramidBackbone(config.blocks)
    raise NotImplementedError("Wrong backbone type")
