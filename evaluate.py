"""
Main evaluating file.
calls the evaluate pipeline with configs.
"""
import argparse

import torch
import wandb
from common.evaluate_pipeline.evaluate import evaluate
from train import get_config
from common.util.logger import logger
from common.util.enums import EnvironmentType

# python train.py --config="grapher_12_conv_gelu_config" --wandb-run-name="grapher only"

parser = argparse.ArgumentParser(
    description="Evaluation Config",
    add_help=True,
)

parser.add_argument(
    "-c",
    "--config",
    default=None,
    type=str,
    help="Give model config.",
)

parser.add_argument(
    "--model-path",
    default=None,
    type=str,
    help="Give model path for evaluation.",
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
    "--dataset",
    type=str,
    default="internal_301_db",
    help="Dataset name.",
)

parser.add_argument(
    "--n-classes",
    type=int,
    default=301,
    help="Defines total classes to predict.",
)

parser.add_argument(
    "--act",
    type=str,
    default="gelu",
    help="Activation type",
)

parser.add_argument(
    "--pred-type",
    type=str,
    default="conv",
    help="Predictor type",
)

parser.add_argument(
    "--height",
    type=int,
    default=224,
    help="Defines height of the image.",
)

parser.add_argument(
    "--width",
    type=int,
    default=224,
    help="Defines width of the image.",
)


def main():
    """
    Wrapper for the driver.
    """
    args = parser.parse_args()
    batch_size = args.batch_size
    logger.info("BATCHSIZE: %s", batch_size)
    environment = (
        EnvironmentType.PYTORCH
        if args.environment == "pytorch"
        else EnvironmentType.TENSORFLOW
    )
    config = get_config(
        args.config,
        args.act,
        args.pred_type,
        args.n_classes,
        2,
        args.height,
        args.width,
    )
    try:
        evaluate(
            args.dataset,
            args.model_path,
            config,
            batch_size,
            environment,
            args.n_classes,
            args.height,
            args.width,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
