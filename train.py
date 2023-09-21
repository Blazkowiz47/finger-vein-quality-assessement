"""
Main training file.
calls the train pipeline with configs.
"""
import argparse

import torch
import wandb
from common_configs import (
    grapher_12_conv_gelu_config,
    grapher_6_conv_gelu_config,
    resnet50_grapher12_conv_gelu_config,
    resnet50_grapher_attention_12_conv_gelu_config,
    grapher_attention_12_conv_gelu_config,
    vig_attention_pyramid_tiny,
    vig_pyramid_tiny,
    vig_stem_grapher12_conv_relu_config,
    vig_stem_grapher_attention_12_conv_gelu_sigmoid,
    vig_stem_grapher_attention_12_conv_relu_config,
)


from common.train_pipeline.train import train
from common.util.logger import logger
from common.util.enums import EnvironmentType

# python train.py --config="grapher_12_conv_gelu_config" --wandb-run-name="grapher only"

parser = argparse.ArgumentParser(
    description="Training Config",
    add_help=True,
)
parser.add_argument(
    "-c",
    "--config",
    default="resnet50_grapher_attention_12_conv_gelu_config",
    type=str,
    help="Put model config name from common_config",
)
parser.add_argument(
    "--epochs",
    default=10_000,
    type=int,
    help="Add number of epochs.",
)
parser.add_argument(
    "--batch-size",
    default=16,
    type=int,
    help="Add batch_size.",
)
parser.add_argument(
    "--environment",
    default="pytorch",
    type=str,
    help="Specify environment. pytorch or tensorflow.",
)
parser.add_argument(
    "--log-on-wandb",
    type=bool,
    default=False,
    help="Whether to log on wandb or not.",
)
parser.add_argument(
    "--wandb-run-name",
    type=str,
    default=None,
    help="Wandb run name",
)
parser.add_argument(
    "--validate-after-epochs",
    type=int,
    default=5,
    help="Validate after epochs.",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-3,
    help="Learning rate.",
)

parser.add_argument(
    "--continue-model",
    type=str,
    default=None,
    help="Give path to the model to continue learning.",
)

parser.add_argument(
    "--augment-times",
    type=int,
    default=0,
    help="Number of augmented images per image",
)


def get_config(config: str):
    """
    Fetches appropriate config.
    """
    if config == "resnet50_grapher12_conv_gelu_config":
        return resnet50_grapher12_conv_gelu_config()
    if config == "resnet50_grapher_attention_12_conv_gelu_config":
        return resnet50_grapher_attention_12_conv_gelu_config()
    if config == "grapher_attention_12_conv_gelu_config":
        return grapher_attention_12_conv_gelu_config()
    if config == "grapher_12_conv_gelu_config":
        return grapher_12_conv_gelu_config()
    if config == "grapher_6_conv_gelu_config":
        return grapher_6_conv_gelu_config()
    if config == "vig_stem_grapher12_conv_relu_config":
        return vig_stem_grapher12_conv_relu_config()
    if config == "vig_stem_grapher_attention_12_conv_relu_config":
        return vig_stem_grapher_attention_12_conv_relu_config()
    if config == "vig_stem_grapher_attention_12_conv_gelu_sigmoid":
        return vig_stem_grapher_attention_12_conv_gelu_sigmoid()
    if config == "vig_pyramid_tiny":
        return vig_pyramid_tiny()
    if config == "vig_attention_pyramid_tiny":
        return vig_attention_pyramid_tiny()
    raise ValueError(f"Wrong config: {config}")


def main():
    """
    Wrapper for the driver.
    """
    args = parser.parse_args()
    log_on_wandb = args.log_on_wandb
    epochs = args.epochs
    batch_size = args.batch_size
    logger.info("BATCHSIZE: %s", batch_size)
    environment = (
        EnvironmentType.PYTORCH
        if args.environment == "pytorch"
        else EnvironmentType.TENSORFLOW
    )
    wandb_run_name = args.wandb_run_name
    config = get_config(args.config)
    if wandb_run_name:
        wandb.init(
            # set the wandb project where this run will be logged
            project="finger-vein-recognition",
            name=wandb_run_name,
            config={
                "architecture": args.config,
                "dataset": "Internal.",
                "epochs": epochs,
            },
        )
    try:
        train(
            config,
            batch_size,
            epochs,
            environment,
            wandb_run_name,
            args.validate_after_epochs,
            args.learning_rate,
            args.continue_model,
            args.augment_times,
        )
    except KeyboardInterrupt:
        pass

    if log_on_wandb:
        wandb.finish()
        torch.cuda.empty_cache()


main()
