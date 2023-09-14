from common.train_pipeline.config import ModelConfiguration


resnet50_grapher12_conv_gelu_config = ModelConfiguration(
    wandb_run_name="resnet50_grapher_conv",
    stem_config={"path": "models/resnet50_pretrained.pt"},
    input_shape=(3, 60, 120),
    backbone_config={
        "pretrained_model": None,
        "act": "gelu",
        "bias": True,
        "n_blocks": 12,
        "in_shape": (1024, 4, 8),
        "channels": 1024,
        "norm": "batch",
        "conv": "mr",
        "stochastic": False,
        "num_knn": 9,
        "use_dilation": False,
        "epsilon": 0.2,
        "drop_path": 0.0,
    },
    predictor_config={
        "pretrained_model": None,
        "type": "conv",
        "act": "gelu",
        "bias": True,
        "in_shape": (1024, 4, 8),
        "channels": 1024,
        "n_classes": 600,
        "hidden_channels": 2048,
    },
)

resnet50_grapher12_linear_gelu_config = ModelConfiguration(
    wandb_run_name="resnet50_grapher_conv",
    stem_config={"path": "models/resnet50_pretrained.pt"},
    input_shape=(3, 60, 120),
    backbone_config={
        "pretrained_model": None,
        "act": "gelu",
        "bias": True,
        "n_blocks": 12,
        "in_shape": (1024, 4, 8),
        "channels": 1024,
        "norm": "batch",
        "conv": "mr",
        "stochastic": False,
        "num_knn": 9,
        "use_dilation": False,
        "epsilon": 0.2,
        "drop_path": 0.0,
    },
    predictor_config={
        "pretrained_model": None,
        "type": "linear",
        "act": "gelu",
        "bias": True,
        "in_shape": (1024, 4, 8),
        "channels": 1024,
        "n_classes": 600,
        "hidden_channels": 2048,
    },
)
