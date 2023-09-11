"""
Trains everything
"""
from sklearn.metrics import confusion_matrix
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from common.data_pipeline.MMCBNU_6000.dataset import DatasetLoader as mmcbnu
from common.data_pipeline.FV_USM.dataset import DatasetLoader as fvusm
from common.util.data_pipeline.dataset_chainer import DatasetChainer
from common.util.enums import EnvironmentType
from common.train_pipeline.isotropic_vig import isotropic_vig_ti_224_gelu


# To watch nvidia-smi continuously after every 2 seconds: watch -n 2 nvidia-smi

BATCH_SIZE = 20
EPOCHS = 20
ENVIRONMENT = EnvironmentType.PYTORCH


def get_model(device: str = "cpu"):
    model = isotropic_vig_ti_224_gelu()
    model.to(device)
    print()
    return model


def get_dataset(environment: EnvironmentType = EnvironmentType.PYTORCH, batch_size: int = 10):
    datasets = DatasetChainer(
        datasets=[
            mmcbnu(included_portion=1, environment_type=environment),
            fvusm(included_portion=0, environment_type=environment),
        ]
    )
    return datasets.get_dataset(environment, batch_size=BATCH_SIZE)


def get_train_loss(device: str = "cpu"):
    if device == "cuda":
        return CrossEntropyLoss().cuda()
    else:
        return CrossEntropyLoss()


def get_test_loss(device: str = "cpu"):
    if device == "cuda":
        return CrossEntropyLoss().cuda()
    else:
        return CrossEntropyLoss()


def get_val_loss(device: str = "cpu"):
    if device == "cuda":
        return CrossEntropyLoss().cuda()
    else:
        return CrossEntropyLoss()


def train(batch_size=10, epochs=1):
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

    model = get_model(device)
    print(model)
    train, test, validation = get_dataset(ENVIRONMENT, BATCH_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loss_fn = get_train_loss(device)
    validate_loss_fn = get_val_loss(device)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in tqdm(train, desc=f"Epoch {epoch}: "):
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = train_loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in validation:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                outputs = model(inputs)
                val_loss += validate_loss_fn(outputs, labels)
                total += labels.size(0)
                predicted = (outputs == outputs.max()).float()
                predicted = predicted.to("cpu").numpy()
                labels = labels.to("cpu").numpy()
                for label, p in zip(labels, predicted):
                    for la, x in zip(label, p):
                        if x == 1.0 and la == x:
                            correct += 1
                            break

        print(
            f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Accuracy: {(correct/total) *100:.5f}%"
        )
    model.train()
