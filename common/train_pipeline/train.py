"""
Trains everything
"""
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from common.data_pipeline.MMCBNU_6000.dataset import DatasetLoader as mmcbnu
from common.data_pipeline.FV_USM.dataset import DatasetLoader as fvusm
from common.util.data_pipeline.dataset_chainer import DatasetChainer
from common.util.enums import EnvironmentType
from common.train_pipeline.isotropic_vig import isotropic_vig_ti_224_gelu


# To watch nvidia-smi continuously after every 2 seconds: watch -n 2 nvidia-smi


def get_model(device: str = "cpu"):
    """
    Gives back a predefined model, sepcified in the config.
    """
    model = isotropic_vig_ti_224_gelu()
    model.to(device)
    print()
    return model


def get_dataset(
    environment: EnvironmentType = EnvironmentType.PYTORCH,
    batch_size: int = 10,
):
    """
    Get's specific dataset within the provided environment.
    Change the datasets using config.
    """
    datasets = DatasetChainer(
        datasets=[
            mmcbnu(
                included_portion=1,
                environment_type=environment,
                train_size=0.85,
                validation_size=0.15,
            ),
            fvusm(included_portion=0, environment_type=environment),
        ]
    )
    return datasets.get_dataset(environment, batch_size=batch_size)


def get_train_loss(device: str = "cpu"):
    """
    Gets a loss function for training.
    """
    if device == "cuda":
        return CrossEntropyLoss().cuda()
    return CrossEntropyLoss()


def get_test_loss(device: str = "cpu"):
    """
    Gets a loss function for training.
    """
    if device == "cuda":
        return CrossEntropyLoss().cuda()
    return CrossEntropyLoss()


def get_val_loss(device: str = "cpu"):
    """
    Gets a loss function for validation.
    """
    if device == "cuda":
        return CrossEntropyLoss().cuda()
    return CrossEntropyLoss()


def train(
    batch_size: int = 10,
    epochs: int = 1,
    environment: EnvironmentType = EnvironmentType.PYTORCH,
):
    """
    Contains the training loop.
    """
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
    train_dataset, _, validation_dataset = get_dataset(environment, batch_size)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loss_fn = get_train_loss(device)
    validate_loss_fn = get_val_loss(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_total = 0
        train_correct = 0
        for inputs, labels in tqdm(train_dataset, desc=f"Epoch {epoch}: "):
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = train_loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            predicted = (outputs == outputs.max()).float()
            predicted = predicted.to("cpu").numpy()
            labels = labels.to("cpu").numpy()
            for label, pred in zip(labels, predicted):
                for class_label, predicted_class in zip(label, pred):
                    if predicted_class == 1.0 and class_label == predicted_class:
                        train_correct += 1
                        break
            train_total += labels.shape[0]

        model.eval()
        val_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in validation_dataset:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                outputs = model(inputs)
                val_loss += validate_loss_fn(outputs, labels)
                total += labels.shape[0]
                predicted = (outputs == outputs.max()).float()
                predicted = predicted.to("cpu").numpy()
                labels = labels.to("cpu").numpy()
                for label, pred in zip(labels, predicted):
                    for class_label, predicted_class in zip(label, pred):
                        if predicted_class == 1.0 and class_label == predicted_class:
                            correct += 1
                            break

        print(
            f"Epoch [{epoch+1}/{epochs}],",
            f"Loss: {loss.item():.4f},",
            f"Val Loss: {val_loss.item():.4f},",
            f"Train Correct: {train_correct},",
            f"Train Total: {train_total},",
            f"Val Correct: {correct},",
            f"Val Total: {total}",
        )
    model.train()
