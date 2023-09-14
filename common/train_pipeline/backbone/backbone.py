"""
Factory for backbones.
"""
import torch
from common.train_pipeline.backbone.isotropic_backbone import IsotropicBackBone
from common.train_pipeline.config import ModelConfiguration


def get_backbone(config: ModelConfiguration):
    """
    Calls appropriate backbone build.
    """
    config = config.backbone_config
    if config["name"] == "isotropic_backbone":
        k = config["num_knn"]
        n_blocks = config["n_blocks"]
        drop_path = config["drop_path"]
        num_knn = [int(x.item()) for x in torch.linspace(k, 2 * k, n_blocks)]
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]
        max_dilation = 196 // max(num_knn)

        return IsotropicBackBone(
            n_blocks=config["n_blocks"],
            channels=config["channels"],
            act=config["act"],
            norm=config["norm"],
            conv=config["conv"],
            stochastic=config["stochastic"],
            num_knn=num_knn,
            use_dilation=config["use_dilation"],
            epsilon=config["epsilon"],
            bias=config["bias"],
            dpr=dpr,
            max_dilation=max_dilation,
        )
