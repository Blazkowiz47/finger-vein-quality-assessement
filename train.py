"""
Main training file.
calls the train pipeline with configs.
"""
import wandb
from common.train_pipeline.model.common_configs import (
    resnet50_grapher12_conv_gelu_config,
    resnet50_grapher_attention_12_conv_gelu_config,
)

from common.train_pipeline.train import train
from common.util.enums import EnvironmentType

BATCH_SIZE = 10
EPOCHS = 1000
ENVIRONMENT = EnvironmentType.PYTORCH
LOG_ON_WANDB = False
RUN_NAME = "600-classes-without-augmentation"


def main():
    """
    Wrapper for the driver.
    """
    if LOG_ON_WANDB:
        wandb.init(
            # set the wandb project where this run will be logged
            project="finger-vein-recognition",
            name=RUN_NAME,
            config={
                "learning_rate": 0.0001,
                "architecture": "ViG with self Attention",
                "dataset": "CIFAR-100",
                "epochs": EPOCHS,
            },
        )
    config = resnet50_grapher_attention_12_conv_gelu_config()
    train(config, BATCH_SIZE, EPOCHS, ENVIRONMENT, LOG_ON_WANDB)
    if LOG_ON_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
