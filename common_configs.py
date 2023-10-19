"""
Common model configs.
"""
from typing import Any, List
import torch
from common.gcn_lib.torch_vertex import GrapherConfig
from common.train_pipeline.backbone.attention_block import AttentionBlockConfig
from common.train_pipeline.backbone.ffn import FFNConfig
from common.train_pipeline.backbone.isotropic_backbone import IsotropicBlockConfig
from common.train_pipeline.backbone.pyramid_backbone import PyramidBlockConfig
from common.train_pipeline.config import (
    BackboneBlockConfig,
    BackboneConfig,
    ModelConfig,
)
from common.train_pipeline.predictor.predictor import PredictorConfig
from common.train_pipeline.stem.stem import StemConfig
from common.util.logger import logger


def pretrained_vig_pyramid_tiny(
    act: str,
    pred_type: str,
    n_classes: int,
    height: int,
    width: int,
) -> ModelConfig:
    """
    Module architecture:
    Resnet50 till layer 3 (Output 1024*4*8) [FROZEN]
    Grapher followed by ffn [12 blocks]
    predictor (linear)
    """
    channels: List[int] = [48, 96, 240, 384]
    num_of_grapher_units: List[int] = [2, 2, 6, 2]
    num_knn: int = 9
    drop_path: float = 0.0
    bias: bool = True
    epsilon: float = 0.2
    conv: str = "mr"
    reduce_ratios: List[int] = [4, 2, 1, 1]

    max_dilation = 196 // num_knn
    blocks: List[PyramidBlockConfig] = []
    original_height, original_width = height, width

    for i in range(4):
        blocks.append(
            PyramidBlockConfig(
                in_channels=channels[i],
                out_channels=channels[i + 1] if i + 1 < len(channels) else channels[i],
                hidden_dimensions_in_ratio=4,
                number_of_grapher_ffn_units=num_of_grapher_units[i],
                grapher_config=GrapherConfig(
                    in_channels=channels[i],
                    act=act,
                    conv=conv,
                    norm="batch",
                    epsilon=epsilon,
                    neighbour_number=min(num_knn, width * height),
                    drop_path=drop_path,
                    max_dilation=max_dilation,
                    dilation=1.0,
                    bias=bias,
                    r=reduce_ratios[i],
                    n=height * width,
                ),
                ffn_config=FFNConfig(
                    channels[i],
                    hidden_features=channels[i] * 4,
                    act=act,
                    drop_path=drop_path,
                    bias=bias,
                ),
            )
        )
        height = height // 4
        width = width // 4

    return ModelConfig(
        height=original_height,
        width=original_width,
        stem_config=StemConfig(
            stem_type="pyramid_3_conv_layer",
            in_channels=3,
            out_channels=channels[0],
            act=act,
            bias=bias,
            requires_grad=False,
        ),
        backbone_config=BackboneConfig(
            backbone_type="pyramid_backbone",
            blocks=blocks,
            requires_grad=False,
        ),
        predictor_config=PredictorConfig(
            predictor_type=pred_type,
            in_channels=channels[-1],
            n_classes=n_classes,
            act=act,
            bias=bias,
            hidden_dims=channels[-1] * 2,
            dropout=0.0,
        ),
    )


def vig_pyramid_tiny(
    act: str,
    pred_type: str,
    n_classes: int,
    height: int,
    width: int,
) -> ModelConfig:
    """
    Module architecture:
    Resnet50 till layer 3 (Output 1024*4*8) [FROZEN]
    Grapher followed by ffn [12 blocks]
    predictor (linear)
    """
    channels: List[int] = [48, 96, 240, 384]
    num_of_grapher_units: List[int] = [2, 2, 6, 2]
    num_knn: int = 9
    drop_path: float = 0.0
    bias: bool = True
    epsilon: float = 0.2
    conv: str = "mr"
    reduce_ratios: List[int] = [4, 2, 1, 1]

    max_dilation = 196 // num_knn
    blocks: List[PyramidBlockConfig] = []
    original_height, original_width = height, width

    for i in range(4):
        blocks.append(
            PyramidBlockConfig(
                in_channels=channels[i],
                out_channels=channels[i + 1] if i + 1 < len(channels) else channels[i],
                hidden_dimensions_in_ratio=4,
                number_of_grapher_ffn_units=num_of_grapher_units[i],
                grapher_config=GrapherConfig(
                    in_channels=channels[i],
                    act=act,
                    conv=conv,
                    norm="batch",
                    epsilon=epsilon,
                    neighbour_number=min(num_knn, width * height),
                    drop_path=drop_path,
                    max_dilation=max_dilation,
                    dilation=1.0,
                    bias=bias,
                    r=reduce_ratios[i],
                    n=height * width,
                ),
                ffn_config=FFNConfig(
                    channels[i],
                    hidden_features=channels[i] * 4,
                    act=act,
                    drop_path=drop_path,
                    bias=bias,
                ),
            )
        )
        height = height // 4
        width = width // 4

    return ModelConfig(
        height=original_height,
        width=original_width,
        stem_config=StemConfig(
            stem_type="pyramid_3_conv_layer",
            in_channels=3,
            out_channels=channels[0],
            act=act,
            bias=bias,
        ),
        backbone_config=BackboneConfig(
            backbone_type="pyramid_backbone",
            blocks=blocks,
        ),
        predictor_config=PredictorConfig(
            predictor_type=pred_type,
            in_channels=channels[-1],
            n_classes=n_classes,
            act=act,
            bias=bias,
            hidden_dims=channels[-1] * 2,
            dropout=0.0,
        ),
    )


def vig_attention_at_last_pyramid_tiny(
    act: str,
    pred_type: str,
    n_classes: int,
    num_heads: int,
    height: int,
    width: int,
) -> ModelConfig:
    """
    Module architecture:
    Resnet50 till layer 3 (Output 1024*4*8) [FROZEN]
    Grapher followed by ffn [12 blocks]
    predictor (linear)
    """
    channels: List[int] = [48, 96, 240, 384]
    num_of_grapher_units: List[int] = [2, 2, 6, 2]
    num_knn: int = 9
    drop_path: float = 0.0
    bias: bool = True
    epsilon: float = 0.2
    conv: str = "mr"
    reduce_ratios: List[int] = [4, 2, 1, 1]

    max_dilation = 196 // num_knn
    original_height, original_width = height, width

    blocks: List[BackboneBlockConfig] = []
    for i in range(4):
        blocks.append(
            PyramidBlockConfig(
                in_channels=channels[i],
                out_channels=channels[i + 1] if i + 1 < len(channels) else channels[i],
                hidden_dimensions_in_ratio=4,
                number_of_grapher_ffn_units=num_of_grapher_units[i],
                grapher_config=GrapherConfig(
                    in_channels=channels[i],
                    act=act,
                    conv=conv,
                    norm="batch",
                    epsilon=epsilon,
                    neighbour_number=min(num_knn, width * height),
                    drop_path=drop_path,
                    max_dilation=max_dilation,
                    dilation=1.0,
                    bias=bias,
                    r=reduce_ratios[i],
                    n=height * width,
                ),
                attention_config=None
                if i < 3
                else AttentionBlockConfig(
                    in_dim=channels[i],
                    num_heads=num_heads,
                    bias=bias,
                    dropout=0,
                ),
                ffn_config=FFNConfig(
                    channels[i],
                    hidden_features=channels[i] * 4,
                    act=act,
                    drop_path=drop_path,
                    bias=bias,
                ),
            )
        )
        height = height // 4
        width = width // 4

    return ModelConfig(
        height=original_height,
        width=original_width,
        stem_config=StemConfig(
            stem_type="pyramid_3_conv_layer",
            in_channels=3,
            out_channels=channels[0],
            act=act,
            bias=bias,
        ),
        backbone_config=BackboneConfig(
            backbone_type="pyramid_backbone",
            blocks=blocks,
        ),
        predictor_config=PredictorConfig(
            predictor_type=pred_type,
            in_channels=channels[-1],
            n_classes=n_classes,
            act=act,
            bias=bias,
            hidden_dims=channels[-1] * 2,
            dropout=0.0,
        ),
    )


def pretrained_vig_attention_only_at_last_pyramid_tiny(
    act: str,
    pred_type: str,
    n_classes: int,
    num_heads: int,
    height: int,
    width: int,
) -> ModelConfig:
    """
    Module architecture:
    Resnet50 till layer 3 (Output 1024*4*8) [FROZEN]
    Grapher followed by ffn [12 blocks]
    predictor (linear)
    """
    channels: List[int] = [48, 96, 240, 384]
    num_of_grapher_units: List[int] = [2, 2, 6, 2]
    num_knn: int = 9
    drop_path: float = 0.0
    bias: bool = True
    epsilon: float = 0.2
    conv: str = "mr"
    reduce_ratios: List[int] = [4, 2, 1, 1]

    max_dilation = 196 // num_knn
    original_height, original_width = height, width

    blocks: List[BackboneBlockConfig] = []
    for i in range(4):
        blocks.append(
            PyramidBlockConfig(
                in_channels=channels[i],
                out_channels=channels[i + 1] if i + 1 < len(channels) else channels[i],
                hidden_dimensions_in_ratio=4,
                number_of_grapher_ffn_units=num_of_grapher_units[i],
                grapher_config=GrapherConfig(
                    in_channels=channels[i],
                    act=act,
                    conv=conv,
                    norm="batch",
                    epsilon=epsilon,
                    neighbour_number=min(num_knn, width * height),
                    drop_path=drop_path,
                    max_dilation=max_dilation,
                    dilation=1.0,
                    bias=bias,
                    r=reduce_ratios[i],
                    n=height * width,
                )
                if i < 3
                else None,
                attention_config=None
                if i < 3
                else AttentionBlockConfig(
                    in_dim=channels[i],
                    num_heads=num_heads,
                    bias=bias,
                    dropout=0,
                ),
                ffn_config=FFNConfig(
                    channels[i],
                    hidden_features=channels[i] * 4,
                    act=act,
                    drop_path=drop_path,
                    bias=bias,
                ),
            )
        )
        height = height // 4
        width = width // 4

    return ModelConfig(
        height=original_height,
        width=original_width,
        stem_config=StemConfig(
            stem_type="pyramid_3_conv_layer",
            in_channels=3,
            out_channels=channels[0],
            act=act,
            bias=bias,
            requires_grad=False,
        ),
        backbone_config=BackboneConfig(
            backbone_type="pyramid_backbone",
            blocks=blocks,
            requires_grad=False,
        ),
        predictor_config=PredictorConfig(
            predictor_type=pred_type,
            in_channels=channels[-1],
            n_classes=n_classes,
            act=act,
            bias=bias,
            hidden_dims=channels[-1] * 2,
            dropout=0.0,
        ),
    )


def vig_attention_only_at_last_pyramid_tiny(
    act: str,
    pred_type: str,
    n_classes: int,
    num_heads: int,
    height: int,
    width: int,
) -> ModelConfig:
    """
    Module architecture:
    Resnet50 till layer 3 (Output 1024*4*8) [FROZEN]
    Grapher followed by ffn [12 blocks]
    predictor (linear)
    """
    channels: List[int] = [48, 96, 240, 384]
    num_of_grapher_units: List[int] = [2, 2, 6, 2]
    num_knn: int = 9
    drop_path: float = 0.0
    bias: bool = True
    epsilon: float = 0.2
    conv: str = "mr"
    reduce_ratios: List[int] = [4, 2, 1, 1]

    max_dilation = 196 // num_knn
    original_height, original_width = height, width

    blocks: List[BackboneBlockConfig] = []
    for i in range(4):
        blocks.append(
            PyramidBlockConfig(
                in_channels=channels[i],
                out_channels=channels[i + 1] if i + 1 < len(channels) else channels[i],
                hidden_dimensions_in_ratio=4,
                number_of_grapher_ffn_units=num_of_grapher_units[i],
                grapher_config=GrapherConfig(
                    in_channels=channels[i],
                    act=act,
                    conv=conv,
                    norm="batch",
                    epsilon=epsilon,
                    neighbour_number=min(num_knn, width * height),
                    drop_path=drop_path,
                    max_dilation=max_dilation,
                    dilation=1.0,
                    bias=bias,
                    r=reduce_ratios[i],
                    n=height * width,
                )
                if i < 3
                else None,
                attention_config=None
                if i < 3
                else AttentionBlockConfig(
                    in_dim=channels[i],
                    num_heads=num_heads,
                    bias=bias,
                    dropout=0,
                ),
                ffn_config=FFNConfig(
                    channels[i],
                    hidden_features=channels[i] * 4,
                    act=act,
                    drop_path=drop_path,
                    bias=bias,
                ),
            )
        )
        height = height // 4
        width = width // 4

    return ModelConfig(
        height=original_height,
        width=original_width,
        stem_config=StemConfig(
            stem_type="pyramid_3_conv_layer",
            in_channels=3,
            out_channels=channels[0],
            act=act,
            bias=bias,
        ),
        backbone_config=BackboneConfig(
            backbone_type="pyramid_backbone",
            blocks=blocks,
        ),
        predictor_config=PredictorConfig(
            predictor_type=pred_type,
            in_channels=channels[-1],
            n_classes=n_classes,
            act=act,
            bias=bias,
            hidden_dims=channels[-1] * 2,
            dropout=0.0,
        ),
    )


def vig_attention_pyramid_tiny(
    act: str,
    pred_type: str,
    n_classes: int,
    num_heads: int,
    height: int,
    width: int,
) -> ModelConfig:
    """
    Module architecture:
    Resnet50 till layer 3 (Output 1024*4*8) [FROZEN]
    Grapher followed by ffn [12 blocks]
    predictor (linear)
    """
    channels: List[int] = [48, 96, 240, 384]
    num_of_grapher_units: List[int] = [2, 2, 6, 2]
    num_knn: int = 9
    drop_path: float = 0.0
    bias: bool = True
    epsilon: float = 0.2
    conv: str = "mr"
    reduce_ratios: List[int] = [4, 2, 1, 1]

    max_dilation = 196 // num_knn
    original_height, original_width = height, width

    blocks: List[BackboneBlockConfig] = []
    for i in range(4):
        blocks.append(
            PyramidBlockConfig(
                in_channels=channels[i],
                out_channels=channels[i + 1] if i + 1 < len(channels) else channels[i],
                hidden_dimensions_in_ratio=4,
                number_of_grapher_ffn_units=num_of_grapher_units[i],
                grapher_config=GrapherConfig(
                    in_channels=channels[i],
                    act=act,
                    conv=conv,
                    norm="batch",
                    epsilon=epsilon,
                    neighbour_number=min(num_knn, width * height),
                    drop_path=drop_path,
                    max_dilation=max_dilation,
                    dilation=1.0,
                    bias=bias,
                    r=reduce_ratios[i],
                    n=height * width,
                ),
                attention_config=AttentionBlockConfig(
                    in_dim=channels[i],
                    num_heads=num_heads,
                    bias=bias,
                    dropout=0,
                ),
                ffn_config=FFNConfig(
                    channels[i],
                    hidden_features=channels[i] * 4,
                    act=act,
                    drop_path=drop_path,
                    bias=bias,
                ),
            )
        )
        height = height // 4
        width = width // 4

    return ModelConfig(
        height=original_height,
        width=original_width,
        stem_config=StemConfig(
            stem_type="pyramid_3_conv_layer",
            in_channels=3,
            out_channels=channels[0],
            act=act,
            bias=bias,
        ),
        backbone_config=BackboneConfig(
            backbone_type="pyramid_backbone",
            blocks=blocks,
        ),
        predictor_config=PredictorConfig(
            predictor_type=pred_type,
            in_channels=channels[-1],
            n_classes=n_classes,
            act=act,
            bias=bias,
            hidden_dims=channels[-1] * 2,
            dropout=0.0,
        ),
    )


def vig_pyramid_compact(
    act: str,
    pred_type: str,
    n_classes: int,
    height: int,
    width: int,
) -> ModelConfig:
    """
    Module architecture:
    Resnet50 till layer 3 (Output 1024*4*8) [FROZEN]
    Grapher followed by ffn [12 blocks]
    predictor (linear)
    """
    channels: List[int] = [64, 128, 256, 512]
    num_of_grapher_units: List[int] = [1, 1, 1, 1]
    num_knn: int = 9
    drop_path: float = 0.0
    bias: bool = True
    epsilon: float = 0.2
    conv: str = "mr"
    reduce_ratios: List[int] = [4, 2, 1, 1]

    max_dilation = channels[-1] // num_knn
    blocks: List[PyramidBlockConfig] = []
    original_height, original_width = height, width

    for i in range(4):
        blocks.append(
            PyramidBlockConfig(
                in_channels=channels[i],
                out_channels=channels[i + 1] if i + 1 < len(channels) else channels[i],
                hidden_dimensions_in_ratio=4,
                number_of_grapher_ffn_units=num_of_grapher_units[i],
                grapher_config=GrapherConfig(
                    in_channels=channels[i],
                    act=act,
                    conv=conv,
                    norm="batch",
                    epsilon=epsilon,
                    neighbour_number=min(num_knn, width * height),
                    drop_path=drop_path,
                    max_dilation=max_dilation,
                    dilation=1.0,
                    bias=bias,
                    r=reduce_ratios[i],
                    n=height * width,
                ),
                ffn_config=FFNConfig(
                    channels[i],
                    hidden_features=channels[i] * 4,
                    act=act,
                    drop_path=drop_path,
                    bias=bias,
                ),
            )
        )
        height = height // 4
        width = width // 4

    return ModelConfig(
        height=original_height,
        width=original_width,
        stem_config=StemConfig(
            stem_type="pyramid_3_conv_layer",
            in_channels=3,
            out_channels=channels[0],
            act=act,
            bias=bias,
        ),
        backbone_config=BackboneConfig(
            backbone_type="pyramid_backbone",
            blocks=blocks,
        ),
        predictor_config=PredictorConfig(
            predictor_type=pred_type,
            in_channels=channels[-1],
            n_classes=n_classes,
            act=act,
            bias=bias,
            hidden_dims=channels[-1] * 2,
            dropout=0.0,
        ),
    )


def vig_pyramid_compact_wo_ffn(
    act: str,
    pred_type: str,
    n_classes: int,
    height: int,
    width: int,
) -> ModelConfig:
    """
    Module architecture:
    Resnet50 till layer 3 (Output 1024*4*8) [FROZEN]
    Grapher followed by ffn [12 blocks]
    predictor (linear)
    """
    total_layers = 7
    channels: List[int] = [256, 512]
    num_of_grapher_units: List[int] = [1, 1]
    num_knn: int = 9
    drop_path: float = 0.0
    bias: bool = True
    epsilon: float = 0.2
    conv: str = "mr"
    reduce_ratios: List[int] = [2, 1]

    max_dilation = channels[-1] // num_knn
    blocks: List[PyramidBlockConfig] = []
    original_height, original_width = height, width

    for i, channel in enumerate(channels):
        blocks.append(
            PyramidBlockConfig(
                in_channels=channel,
                out_channels=channels[i + 1] if i + 1 < len(channels) else channel,
                hidden_dimensions_in_ratio=4,
                number_of_grapher_ffn_units=num_of_grapher_units[i],
                grapher_config=GrapherConfig(
                    in_channels=channel,
                    act=act,
                    conv=conv,
                    norm="batch",
                    epsilon=epsilon,
                    neighbour_number=min(num_knn, width * height),
                    drop_path=drop_path,
                    max_dilation=max_dilation,
                    dilation=1.0,
                    bias=bias,
                    r=reduce_ratios[i],
                    n=height * width,
                ),
            )
        )
        height = height // 4
        width = width // 4

    return ModelConfig(
        height=original_height,
        width=original_width,
        stem_config=StemConfig(
            stem_type="pyramid_3_conv_layer",
            in_channels=3,
            out_channels=channels[0],
            total_layers=total_layers,
            act=act,
            bias=bias,
        ),
        backbone_config=BackboneConfig(
            backbone_type="pyramid_backbone",
            blocks=blocks,
        ),
        predictor_config=PredictorConfig(
            predictor_type=pred_type,
            in_channels=channels[-1],
            n_classes=n_classes,
            act=act,
            bias=bias,
            hidden_dims=channels[-1] * 2,
            dropout=0.0,
        ),
    )
