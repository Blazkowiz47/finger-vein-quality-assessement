"""
Model Factory.
"""

from torch.nn import Module, Sequential

from common.train_pipeline.config import TrainConfiguration


def get_model(config: TrainConfiguration) -> Module:
    

