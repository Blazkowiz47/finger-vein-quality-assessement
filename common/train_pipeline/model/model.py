"""
Model Factory.
"""

from torch.nn import Module
from common.train_pipeline.config import ModelConfig
from common.train_pipeline.model.custom_model import CustomModel


def get_model(config: ModelConfig) -> Module:
    """
    Gives back appropriate models.
    """
    return CustomModel(config=config)
