"""
Main training file.
calls the train pipeline with configs.
"""
import argparse

import wandb
import yaml
from common_configs import (
    resnet50_grapher12_conv_gelu_config,
    resnet50_grapher_attention_12_conv_gelu_config,
    grapher_attention_12_conv_gelu_config,
)


from common.train_pipeline.train import train
from common.util.enums import EnvironmentType


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
    default=10,
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

    raise ValueError(f"Wrong config: {config}")


def main():
    """
    Wrapper for the driver.
    """
    args = parser.parse_args()
    log_on_wandb = args.log_on_wandb
    epochs = args.epochs
    batch_size = args.batch_size
    environment = (
        EnvironmentType.PYTORCH
        if args.environment == "pytorch"
        else EnvironmentType.TENSORFLOW
    )
    wandb_run_name = args.wandb_run_name
    config = get_config(args.config)
    print(log_on_wandb, epochs, batch_size, environment, wandb_run_name, args.config)
    if log_on_wandb:
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

    train(config, batch_size, epochs, environment, log_on_wandb)
    if log_on_wandb:
        wandb.finish()


main()
