"""
Main training file.
calls the train pipeline with configs.
"""
import argparse

import torch
import wandb
import common_configs as cfgs
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

parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    help="Dataset name.",
)

parser.add_argument(
    "--act",
    type=str,
    default="gelu",
    help="Defines activation function.",
)

parser.add_argument(
    "--pred-type",
    type=str,
    default="conv",
    help="Defines predictor type.",
)

parser.add_argument(
    "--n-classes",
    type=int,
    default=301,
    help="Defines total classes to predict.",
)


parser.add_argument(
    "--num-heads",
    type=int,
    default=4,
    help="Defines total number of heads in attention.",
)

parser.add_argument(
    "--height",
    type=int,
    default=60,
    help="Defines height of the image.",
)

parser.add_argument(
    "--width",
    type=int,
    default=120,
    help="Defines width of the image.",
)


def get_config(
    config: str,
    act: str,
    pred_type: str,
    n_classes: int,
    num_heads: int,
    height: int,
    width: int,
):
    """
    Fetches appropriate config.
    """
    if config == "resnet50_grapher_12_conv_config":
        return cfgs.resnet50_grapher_12_conv_config(
            act,
            pred_type,
            n_classes,
            height,
            width,
        )
    if config == "resnet50_grapher_attention_12_conv_config":
        return cfgs.resnet50_grapher_attention_12_conv_config(
            act,
            pred_type,
            n_classes,
            height,
            width,
        )
    if config == "grapher_6_conv_config":
        return cfgs.grapher_6_conv_config(
            act,
            pred_type,
            n_classes,
            height,
            width,
        )
    if config == "grapher_12_conv_config":
        return cfgs.grapher_12_conv_config(
            act,
            pred_type,
            n_classes,
            height,
            width,
        )
    if config == "vig_pyramid_tiny":
        return cfgs.vig_pyramid_tiny(
            act,
            pred_type,
            n_classes,
            height,
            width,
        )

    if config == "pretrained_vig_pyramid_tiny":
        return cfgs.pretrained_vig_pyramid_tiny(
            act,
            pred_type,
            n_classes,
            height,
            width,
        )
    if config == "vig_attention_at_last_pyramid_tiny":
        return cfgs.vig_attention_at_last_pyramid_tiny(
            act,
            pred_type,
            n_classes,
            num_heads,
            height,
            width,
        )

    if config == "vig_attention_pyramid_tiny":
        return cfgs.vig_attention_pyramid_tiny(
            act,
            pred_type,
            n_classes,
            num_heads,
            height,
            width,
        )

    if config == "vig_attention_only_at_last_pyramid_tiny":
        return cfgs.vig_attention_only_at_last_pyramid_tiny(
            act,
            pred_type,
            n_classes,
            num_heads,
            height,
            width,
        )

    if config == "pretrained_vig_attention_only_at_last_pyramid_tiny":
        return cfgs.pretrained_vig_attention_only_at_last_pyramid_tiny(
            act,
            pred_type,
            n_classes,
            num_heads,
            height,
            width,
        )

    raise ValueError(f"Wrong config: {config}")


def main():
    """
    Wrapper for the driver.
    """
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    logger.info("BATCHSIZE: %s", batch_size)
    environment = (
        EnvironmentType.PYTORCH
        if args.environment == "pytorch"
        else EnvironmentType.TENSORFLOW
    )
    wandb_run_name = args.wandb_run_name
    config = get_config(
        args.config,
        args.act,
        args.pred_type,
        args.n_classes,
        args.num_heads,
        args.height,
        args.width,
    )
    if wandb_run_name:
        wandb.init(
            # set the wandb project where this run will be logged
            project="finger-vein-recognition",
            name=wandb_run_name,
            config={
                "architecture": args.config,
                "dataset": args.dataset,
                "epochs": epochs,
                "activation": args.act,
                "predictor_type": args.pred_type,
            },
        )
    try:
        train(
            config,
            args.dataset,
            batch_size,
            epochs,
            environment,
            wandb_run_name,
            args.validate_after_epochs,
            args.learning_rate,
            args.continue_model,
            args.augment_times,
            args.n_classes,
            args.height,
            args.width,
        )
    except KeyboardInterrupt:
        pass

    if wandb_run_name:
        wandb.finish()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
