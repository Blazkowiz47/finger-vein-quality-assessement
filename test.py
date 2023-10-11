import torch
from torchviz import make_dot

from train import get_config
from common.train_pipeline.model.model import get_model


config = get_config(
    "vig_attention_at_last_pyramid_tiny", "gelu", "conv", 2, 2, 224, 224
)
model = get_model(config)
print(model)
# x = torch.randn(2, 3, 224, 224)
# y = model(x)
# make_dot(y, params=dict(model.named_parameters())).render(
#    "vig_attention_at_last_pyramid_tiny", format="png"
# )

import cv2 as cv2

# img = cv2.imread("vig_attention_at_last_pyramid_tiny.png", cv2.IMREAD_COLOR)
# print(img.shape)

# cv2.imshow("asd", img[:1000, :1000, :])
# cv2.waitKey(0)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
