from common.train_pipeline.train import train

BATCH_SIZE = 20
EPOCHS = 20


def main():
    train(BATCH_SIZE, EPOCHS)


if __name__ == "__main__":
    main()
