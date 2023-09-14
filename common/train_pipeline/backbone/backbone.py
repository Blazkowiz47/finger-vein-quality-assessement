"""
Factory for backbones.
"""

from common.train_pipeline.config import TrainConfiguration


def get_backbone(config: TrainConfiguration):
    if config.backbone == "isotropic_backbone":
        return 