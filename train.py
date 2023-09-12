"""
Main training file.
calls the train pipeline with configs.
"""
from common.train_pipeline.train import train
from common.util.enums import EnvironmentType

BATCH_SIZE = 20
EPOCHS = 20
ENVIRONMENT = EnvironmentType.PYTORCH


def main():
    """
    Wrapper for the driver.
    """
    train(BATCH_SIZE, EPOCHS)


if __name__ == "__main__":
    main()
