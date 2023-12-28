"""
Main training file.
calls the train pipeline with configs.
"""
import argparse

import torch
import wandb
from common.train_pipeline.config import ModelConfig
import common_configs as cfgs
from common.train_pipeline.train import train
from common.train_pipeline.arcvein_train import train as arc_train
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
    default="vig_pyramid_compact",
    type=str,
    help="Put model config name from common_config",
)
parser.add_argument(
    "--epochs",
    default=30,
    type=int,
    help="Add number of epochs.",
)
parser.add_argument(
    "--batch-size",
    default=32,
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
    default=1,
    help="Validate after epochs.",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-4,
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
    default=4,
    help="Number of augmented images per image",
)

parser.add_argument(
    "--dataset",
    type=str,
    default="internal_301_db",
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
    default=224,
    help="Defines height of the image.",
)

parser.add_argument(
    "--width",
    type=int,
    default=224,
    help="Defines width of the image.",
)
parser.add_argument(
    "--pretrained-model-path",
    type=str,
    default=None,
    help="Defines pretrained model's path.",
)

parser.add_argument(
    "--pretrained-classes",
    type=int,
    default=None,
    help="Defines pretrained model's predictor_classes.",
)

parser.add_argument(
    "--total-layers",
    type=int,
    default=5,
    help="TOTAL LAYERS FOR DSC STEM",
)


parser.add_argument(
    "--grapher-units",
    type=str,
    default="2,1,6,2",
    help="Number of grapher units",
)


def get_config(
    config: str,
    act: str,
    pred_type: str,
    n_classes: int,
    num_heads: int,
    height: int,
    width: int,
    total_layers: int,
    grapher_units: str,
) -> ModelConfig:
    """
    Fetches appropriate config.
    """
    if config == "arcvein":
        cfg = ModelConfig()
        cfg.arcvein = True
        return cfg

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

    if config == "vig_pyramid_compact":
        return cfgs.vig_pyramid_compact(
            act,
            pred_type,
            n_classes,
            height,
            width,
        )

    if config == "test_vig_custom":
        return cfgs.test_vig_custom(
            act,
            pred_type,
            n_classes,
            height,
            width,
        )

    if config == "test_dsc_custom":
        graphers = [int(x) for x in grapher_units.split(",")]
        return cfgs.test_dsc_custom(
            act,
            pred_type,
            n_classes,
            height,
            width,
            total_layers,
            grapher_units=graphers,
        )
    if config == "test_dsc_wo_grapher":
        return cfgs.test_dsc_wo_grapher(
            act,
            pred_type,
            n_classes,
            height,
            width,
            total_layers,
        )

    if config == "test_wo_dsc_wo_grapher":
        return cfgs.test_wo_dsc_wo_grapher(
            act,
            pred_type,
            n_classes,
            height,
            width,
            total_layers,
        )

    if config == "test_wo_dsc_custom":
        graphers = [int(x) for x in grapher_units.split(",")]
        return cfgs.test_wo_dsc_custom(
            act,
            pred_type,
            n_classes,
            height,
            width,
            total_layers,
            grapher_units=graphers,
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
        args.total_layers,
        args.grapher_units,
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
        if args.config == "arcvein":
            arc_train(
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
                pretrained_model_path=args.pretrained_model_path,
                pretrained_predictor_classes=args.pretrained_classes,
            )
        else:
            eer, tar1, tar01, tar001 = train(
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
                pretrained_model_path=args.pretrained_model_path,
                pretrained_predictor_classes=args.pretrained_classes,
            )
            import json

            with open(f"results/{wandb_run_name}.json", "w+") as fp:
                json.dump(
                    {
                        "eer": eer,
                        "1": tar1,
                        "0.1": tar01,
                        "0.01": tar001,
                    },
                    fp,
                )
    except KeyboardInterrupt:
        pass

    if wandb_run_name:
        wandb.finish()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
