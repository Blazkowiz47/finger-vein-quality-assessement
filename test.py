import torch
from torchviz import make_dot

from train import get_config
from common.train_pipeline.model.model import get_model


config = get_config(
    "vig_attention_at_last_pyramid_tiny", "gelu", "conv", 2, 2, 224, 224
)
model = get_model(config)
x = torch.randn(2, 3, 224, 224)
y = model(x)
make_dot(y, params=dict(model.named_parameters()))
