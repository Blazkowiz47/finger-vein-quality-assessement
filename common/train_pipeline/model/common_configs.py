"""
Common model configs.
"""
from typing import Any, List
import torch
from common.gcn_lib.torch_vertex import GrapherConfig
from common.train_pipeline.backbone.ffn import FFNConfig
from common.train_pipeline.backbone.isotropic_backbone import IsotropicBlockConfig
from common.train_pipeline.config import BackboneConfig, ModelConfig
from common.train_pipeline.predictor.predictor import PredictorConfig
from common.train_pipeline.stem.stem import StemConfig


def resnet50_grapher12_conv_gelu_config() -> ModelConfig:
    """
    Module architecture:
    Resnet50 till layer 3 (Output 1024*4*8) [FROZEN]
    Grapher followed by ffn [12 blocks]
    predictor (linear)
    """
    in_channels: int = 1024
    act: str = "gelu"
    n_blocks: int = 12
    num_knn: int = 9
    drop_path: float = 0.0
    use_dilation: bool = False
    n_classes: int = 600
    bias: bool = True

    num_knn_list: List[Any] = [
        int(x.item()) for x in torch.linspace(num_knn, 2 * num_knn, n_blocks)
    ]
    dpr = [
        x.item() for x in torch.linspace(0, drop_path, n_blocks)
    ]  # stochastic depth decay rule
    print("dpr", dpr)
    print("Num knn:", num_knn_list)
    max_dilation = 196 // max(num_knn_list)
    blocks: List[IsotropicBlockConfig] = []

    for index in range(n_blocks):
        blocks.append(
            IsotropicBlockConfig(
                block_type="grapher_ffn",
                grapher_config=GrapherConfig(
                    in_channels=in_channels,
                    act=act,
                    norm="batch",
                    epsilon=0.002,
                    neighbour_number=num_knn_list[index],
                    drop_path=dpr[index],
                    max_dilation=max_dilation,
                    dilation=min(index // 4 + 1, max_dilation) if use_dilation else 1.0,
                    bias=bias,
                ),
                ffn_config=FFNConfig(
                    in_channels,
                    hidden_features=2048,
                    act=act,
                    drop_path=0.0,
                    bias=bias,
                ),
            )
        )

    return ModelConfig(
        stem_config=StemConfig(
            stem_type="pretrained_resnet50",
            pretrained="models/resnet50_pretrained.pt",
            resnet_layer=3,
            requires_grad=False,
            bias=bias,
        ),
        backbone_config=BackboneConfig(
            backbone_type="isotropic_backbone",
            blocks=blocks,
        ),
        predictor_config=PredictorConfig(
            predictor_type="linear",
            in_channels=in_channels,
            linear_dims=in_channels * 3,
            n_classes=n_classes,
            act=act,
            bias=bias,
            hidden_dims=2048,
            dropout=0.0,
            conv_out_channels=in_channels // 4,
        ),
    )
