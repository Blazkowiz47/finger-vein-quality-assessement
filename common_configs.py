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
from common.train_pipeline.config import BackboneConfig, ModelConfig
from common.train_pipeline.predictor.predictor import PredictorConfig
from common.train_pipeline.stem.stem import StemConfig
from common.util.logger import logger

# For 60*120 backbone output is 1024*4*8
# For 100*300 backbone output is 1024*7*19
# For 100*200 backbone output is 1024*7*13


def resnet50_grapher_12_conv_config(act: str, pred_type: str) -> ModelConfig:
    """
    Module architecture:
    Resnet50 till layer 3 (Output 1024*4*8) [FROZEN]
    Grapher followed by ffn [12 blocks]
    predictor (linear)
    """
    resnet_layer: int = 3
    in_channels: int = 1024
    linear_dims: int = 24576
    n_blocks: int = 12
    num_knn: int = 9
    drop_path: float = 0.0
    use_dilation: bool = False
    n_classes: int = 301
    bias: bool = True
    epsilon: float = 0.2
    hidden_features: int = 2048
    conv: str = "mr"

    num_knn_list: List[Any] = [
        int(x.item()) for x in torch.linspace(num_knn, 2 * num_knn, n_blocks)
    ]
    dpr = [
        x.item() for x in torch.linspace(0, drop_path, n_blocks)
    ]  # stochastic depth decay rule
    logger.info(f"dpr {dpr}")
    logger.info(f"Num knn: {num_knn_list}")
    max_dilation = 196 // max(num_knn_list)
    blocks: List[IsotropicBlockConfig] = []

    for index in range(n_blocks):
        blocks.append(
            IsotropicBlockConfig(
                block_type="grapher_ffn",
                grapher_config=GrapherConfig(
                    in_channels=in_channels,
                    act=act,
                    conv=conv,
                    norm="batch",
                    epsilon=epsilon,
                    neighbour_number=num_knn_list[index],
                    drop_path=dpr[index],
                    max_dilation=max_dilation,
                    dilation=min(index // 4 + 1, max_dilation) if use_dilation else 1.0,
                    bias=bias,
                ),
                ffn_config=FFNConfig(
                    in_channels,
                    hidden_features=hidden_features,
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
            resnet_layer=resnet_layer,
            requires_grad=False,
            bias=bias,
        ),
        backbone_config=BackboneConfig(
            backbone_type="isotropic_backbone",
            blocks=blocks,
        ),
        predictor_config=PredictorConfig(
            predictor_type=pred_type,
            in_channels=in_channels,
            n_classes=n_classes,
            act=act,
            bias=bias,
            hidden_dims=512,
            dropout=0.0,
        ),
    )


def resnet50_grapher_attention_12_conv_config(act: str, pred_type: str) -> ModelConfig:
    """
    Module architecture:
    Resnet50 till layer 3 (Output 1024*4*8) [FROZEN]
    Grapher followed by ffn [12 blocks]
    predictor (linear)
    """
    resnet_layer: int = 3
    in_channels: int = 1024
    linear_dims: int = 1024
    n_blocks: int = 12
    num_knn: int = 9
    drop_path: float = 0.0
    use_dilation: bool = False
    n_classes: int = 301
    bias: bool = True
    epsilon: float = 0.2
    hidden_features: int = 2048
    conv: str = "mr"

    num_knn_list: List[Any] = [
        int(x.item()) for x in torch.linspace(num_knn, 2 * num_knn, n_blocks)
    ]
    dpr = [
        x.item() for x in torch.linspace(0, drop_path, n_blocks)
    ]  # stochastic depth decay rule
    logger.info(f"dpr {dpr}")
    logger.info(f"Num knn: {num_knn_list}")
    max_dilation = 196 // max(num_knn_list)
    blocks: List[IsotropicBlockConfig] = []

    for index in range(n_blocks):
        blocks.append(
            IsotropicBlockConfig(
                block_type="grapher_attention_ffn",
                grapher_config=GrapherConfig(
                    in_channels=in_channels,
                    act=act,
                    conv=conv,
                    norm="batch",
                    epsilon=epsilon,
                    neighbour_number=num_knn_list[index],
                    drop_path=dpr[index],
                    max_dilation=max_dilation,
                    dilation=min(index // 4 + 1, max_dilation) if use_dilation else 1.0,
                    bias=bias,
                ),
                attention_config=AttentionBlockConfig(
                    in_dim=in_channels,
                    num_heads=4,
                    bias=bias,
                    dropout=0,
                ),
                ffn_config=FFNConfig(
                    in_channels,
                    hidden_features=hidden_features,
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
            resnet_layer=resnet_layer,
            requires_grad=False,
            bias=bias,
        ),
        backbone_config=BackboneConfig(
            backbone_type="isotropic_backbone",
            blocks=blocks,
        ),
        predictor_config=PredictorConfig(
            predictor_type=pred_type,
            in_channels=in_channels,
            n_classes=n_classes,
            act=act,
            bias=bias,
            hidden_dims=512,
            dropout=0.0,
        ),
    )


def grapher_attention_12_conv_config(act: str, pred_type: str) -> ModelConfig:
    """
    Module architecture:
    Resnet50 till layer 3 (Output 1024*4*8) [FROZEN]
    Grapher followed by ffn [12 blocks]
    predictor (linear)
    """
    resnet_layer: int = 3
    in_channels: int = 1024
    linear_dims: int = 24576
    n_blocks: int = 12
    num_knn: int = 9
    drop_path: float = 0.0
    use_dilation: bool = False
    n_classes: int = 301
    bias: bool = True
    epsilon: float = 0.2
    hidden_features: int = 2048
    conv: str = "mr"

    num_knn_list: List[Any] = [
        int(x.item()) for x in torch.linspace(num_knn, 2 * num_knn, n_blocks)
    ]
    dpr = [
        x.item() for x in torch.linspace(0, drop_path, n_blocks)
    ]  # stochastic depth decay rule
    logger.info(f"dpr {dpr}")
    logger.info(f"Num knn: {num_knn_list}")
    max_dilation = 196 // max(num_knn_list)
    blocks: List[IsotropicBlockConfig] = []

    for index in range(n_blocks):
        blocks.append(
            IsotropicBlockConfig(
                block_type="grapher_attention_ffn",
                grapher_config=GrapherConfig(
                    in_channels=in_channels,
                    conv=conv,
                    act=act,
                    norm="batch",
                    epsilon=epsilon,
                    neighbour_number=num_knn_list[index],
                    drop_path=dpr[index],
                    max_dilation=max_dilation,
                    dilation=min(index // 4 + 1, max_dilation) if use_dilation else 1.0,
                    bias=bias,
                ),
                attention_config=AttentionBlockConfig(
                    in_dim=in_channels,
                    num_heads=4,
                    bias=bias,
                    dropout=0,
                ),
                ffn_config=FFNConfig(
                    in_channels,
                    hidden_features=hidden_features,
                    act=act,
                    drop_path=0.0,
                    bias=bias,
                ),
            )
        )

    return ModelConfig(
        backbone_config=BackboneConfig(
            backbone_type="isotropic_backbone",
            blocks=blocks,
        ),
        predictor_config=PredictorConfig(
            predictor_type=pred_type,
            in_channels=in_channels,
            n_classes=n_classes,
            act=act,
            bias=bias,
            hidden_dims=512,
            dropout=0.0,
        ),
    )


def grapher_12_conv_config(act: str, pred_type: str) -> ModelConfig:
    """
    Module architecture:
    Resnet50 till layer 3 (Output 1024*4*8) [FROZEN]
    Grapher followed by ffn [12 blocks]
    predictor (linear)
    """

    resnet_layer: int = 3
    in_channels: int = 1024
    linear_dims: int = 24576
    n_blocks: int = 12
    num_knn: int = 9
    drop_path: float = 0.0
    use_dilation: bool = True
    n_classes: int = 301
    bias: bool = True
    epsilon: float = 0.2
    hidden_features: int = 2048
    conv: str = "mr"

    num_knn_list: List[Any] = [
        int(x.item()) for x in torch.linspace(num_knn, 2 * num_knn, n_blocks)
    ]
    dpr = [
        x.item() for x in torch.linspace(0, drop_path, n_blocks)
    ]  # stochastic depth decay rule
    logger.info(f"dpr {dpr}")
    logger.info(f"Num knn: {num_knn_list}")
    max_dilation = 196 // max(num_knn_list)
    blocks: List[IsotropicBlockConfig] = []

    for index in range(n_blocks):
        blocks.append(
            IsotropicBlockConfig(
                block_type="grapher_ffn",
                grapher_config=GrapherConfig(
                    in_channels=in_channels,
                    act=act,
                    conv=conv,
                    norm="batch",
                    epsilon=epsilon,
                    neighbour_number=num_knn_list[index],
                    drop_path=dpr[index],
                    max_dilation=max_dilation,
                    dilation=min(index // 4 + 1, max_dilation) if use_dilation else 1.0,
                    bias=bias,
                ),
                ffn_config=FFNConfig(
                    in_channels,
                    hidden_features=hidden_features,
                    act=act,
                    drop_path=0.0,
                    bias=bias,
                ),
            )
        )

    return ModelConfig(
        backbone_config=BackboneConfig(
            backbone_type="isotropic_backbone",
            blocks=blocks,
        ),
        predictor_config=PredictorConfig(
            predictor_type=pred_type,
            in_channels=in_channels,
            n_classes=n_classes,
            act=act,
            bias=bias,
            hidden_dims=512,
            dropout=0.0,
        ),
    )


def grapher_6_conv_config(act: str, pred_type: str) -> ModelConfig:
    """
    Module architecture:
    Resnet50 till layer 3 (Output 1024*4*8) [FROZEN]
    Grapher followed by ffn [12 blocks]
    predictor (linear)
    """

    resnet_layer: int = 3
    in_channels: int = 1024
    linear_dims: int = 1024
    n_blocks: int = 6
    num_knn: int = 9
    drop_path: float = 0.0
    use_dilation: bool = True
    n_classes: int = 301
    bias: bool = True
    epsilon: float = 0.2
    hidden_features: int = 2048
    conv: str = "mr"

    num_knn_list: List[Any] = [
        int(x.item()) for x in torch.linspace(num_knn, 2 * num_knn, n_blocks)
    ]
    dpr = [
        x.item() for x in torch.linspace(0, drop_path, n_blocks)
    ]  # stochastic depth decay rule
    logger.info(f"dpr {dpr}")
    logger.info(f"Num knn: {num_knn_list}")
    max_dilation = 196 // max(num_knn_list)
    blocks: List[IsotropicBlockConfig] = []

    for index in range(n_blocks):
        blocks.append(
            IsotropicBlockConfig(
                block_type="grapher_ffn",
                grapher_config=GrapherConfig(
                    in_channels=in_channels,
                    act=act,
                    conv=conv,
                    norm="batch",
                    epsilon=epsilon,
                    neighbour_number=num_knn_list[index],
                    drop_path=dpr[index],
                    max_dilation=max_dilation,
                    dilation=min(index // 4 + 1, max_dilation) if use_dilation else 1.0,
                    bias=bias,
                ),
                ffn_config=FFNConfig(
                    in_channels,
                    hidden_features=hidden_features,
                    act=act,
                    drop_path=0.0,
                    bias=bias,
                ),
            )
        )

    return ModelConfig(
        backbone_config=BackboneConfig(
            backbone_type="isotropic_backbone",
            blocks=blocks,
        ),
        predictor_config=PredictorConfig(
            predictor_type=pred_type,
            in_channels=in_channels,
            n_classes=n_classes,
            act=act,
            bias=bias,
            hidden_dims=512,
            dropout=0.0,
        ),
    )


def vig_pyramid_tiny(act: str, pred_type: str) -> ModelConfig:
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
    use_dilation: bool = True
    n_classes: int = 301
    bias: bool = True
    epsilon: float = 0.2
    conv: str = "mr"
    linear_dims: int = 9216

    num_knn_list: List[Any] = [int(x.item()) for x in torch.linspace(4, 2 * num_knn, 4)]
    num_knn_list.reverse()
    logger.info("KNNs: %s", num_knn_list)
    max_dilation = 196 // num_knn
    H = 128
    W = 256
    blocks: List[PyramidBlockConfig] = []
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
                    neighbour_number=num_knn_list[i],
                    drop_path=drop_path,
                    max_dilation=max_dilation,
                    dilation=1.0,
                    bias=bias,
                    n=H * W,
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
        H = H // 4
        W = W // 4

    return ModelConfig(
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


def vig_attention_pyramid_tiny(act: str, pred_type: str) -> ModelConfig:
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
    use_dilation: bool = True
    n_classes: int = 301
    bias: bool = True
    epsilon: float = 0.2
    conv: str = "mr"

    num_knn_list: List[Any] = [int(x.item()) for x in torch.linspace(4, 2 * num_knn, 4)]
    num_knn_list.reverse()
    logger.info("KNNs: %s", num_knn_list)
    max_dilation = 196 // num_knn
    height = 128
    width = 256
    blocks: List[PyramidBlockConfig] = []
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
                    neighbour_number=num_knn_list[i],
                    drop_path=drop_path,
                    max_dilation=max_dilation,
                    dilation=1.0,
                    bias=bias,
                    n=height * width,
                ),
                attention_config=AttentionBlockConfig(
                    in_dim=channels[i],
                    num_heads=4,
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


def vig_pyramid_tiny_classification(act: str, pred_type: str) -> ModelConfig:
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
    use_dilation: bool = True
    n_classes: int = 2
    bias: bool = True
    epsilon: float = 0.2
    conv: str = "mr"
    linear_dims: int = 384
    reduce_ratios = [4, 2, 1, 1]

    max_dilation = 196 // 9
    height = 224
    width = 224
    blocks: List[PyramidBlockConfig] = []
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
                    neighbour_number=9,
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


def vig_attention_pyramid_tiny_classification(act: str, pred_type: str) -> ModelConfig:
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
    use_dilation: bool = True
    n_classes: int = 2
    bias: bool = True
    epsilon: float = 0.2
    conv: str = "mr"
    linear_dims: int = 19200

    num_knn_list: List[Any] = [int(x.item()) for x in torch.linspace(4, 2 * num_knn, 4)]
    num_knn_list.reverse()
    logger.info("KNNs: %s", num_knn_list)
    max_dilation = 196 // num_knn
    H = 128
    W = 256
    blocks: List[PyramidBlockConfig] = []
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
                    neighbour_number=num_knn_list[i],
                    drop_path=drop_path,
                    max_dilation=max_dilation,
                    dilation=1.0,
                    bias=bias,
                    n=H * W,
                ),
                attention_config=AttentionBlockConfig(
                    in_dim=channels[i],
                    num_heads=4,
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
        H = H // 4
        W = W // 4

    return ModelConfig(
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


def vig_pyramid_tiny_conv_classification(act: str, pred_type: str) -> ModelConfig:
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
    n_classes: int = 2
    bias: bool = True
    epsilon: float = 0.2
    conv: str = "mr"
    linear_dims: int = 19200

    num_knn_list: List[Any] = [int(x.item()) for x in torch.linspace(4, 2 * num_knn, 4)]
    num_knn_list.reverse()
    logger.info("KNNs: %s", num_knn_list)
    max_dilation = 196 // num_knn
    height = 128
    width = 256
    blocks: List[PyramidBlockConfig] = []
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
                    neighbour_number=num_knn_list[i],
                    drop_path=drop_path,
                    max_dilation=max_dilation,
                    dilation=1.0,
                    bias=bias,
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


def vig_attention_pyramid_tiny_conv_classification(
    act: str, pred_type: str
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
    use_dilation: bool = True
    n_classes: int = 2
    bias: bool = True
    epsilon: float = 0.2
    conv: str = "mr"
    linear_dims: int = 19200

    num_knn_list: List[Any] = [int(x.item()) for x in torch.linspace(4, 2 * num_knn, 4)]
    num_knn_list.reverse()
    logger.info("KNNs: %s", num_knn_list)
    max_dilation = 196 // num_knn
    H = 128
    W = 256
    blocks: List[PyramidBlockConfig] = []
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
                    neighbour_number=num_knn_list[i],
                    drop_path=drop_path,
                    max_dilation=max_dilation,
                    dilation=1.0,
                    bias=bias,
                    n=H * W,
                ),
                attention_config=AttentionBlockConfig(
                    in_dim=channels[i],
                    num_heads=4,
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
        H = H // 4
        W = W // 4

    return ModelConfig(
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
