"""
Model Factory.
"""

from torch.nn import Module
from common.train_pipeline.model.isotropic_vig import isotropic_vig_ti_224_gelu


def get_model(model_name: str) -> Module:
    """
    Gives back appropriate models.
    """
    if model_name == "isotropic_vig_ti_224_gelu":
        return isotropic_vig_ti_224_gelu()
