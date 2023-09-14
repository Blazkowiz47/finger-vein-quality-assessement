"""
Main training file.
calls the train pipeline with configs.
"""
import wandb

from common.train_pipeline.train import train
from common.util.enums import EnvironmentType

BATCH_SIZE = 10
EPOCHS = 1000
ENVIRONMENT = EnvironmentType.PYTORCH
LOG_ON_WANDB = True


def main():
    """
    Wrapper for the driver.
    """
    if LOG_ON_WANDB:
        wandb.init(
            # set the wandb project where this run will be logged
            project="finger-vein-recognition",
            name="600-classes-without-augmentation",
            config={
                "learning_rate": 0.0001,
                "architecture": "ViG with self Attention",
                "dataset": "CIFAR-100",
                "epochs": EPOCHS,
            },
        )
    train(BATCH_SIZE, EPOCHS, ENVIRONMENT, LOG_ON_WANDB)
    if LOG_ON_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
