"""
Main explainability file.
calls the evaluate pipeline with configs.
"""
import argparse
from PIL import Image
import torch
from common.util.logger import logger
from common.util.enums import EnvironmentType

from timm.loss import SoftTargetCrossEntropy
from common.train_pipeline.model.model import get_model
from tqdm import tqdm
import numpy as np
from common.data_pipeline.dataset import get_dataset

from train import get_config
from common.train_pipeline.train import cuda_info
from common.util.data_pipeline.dataset_chainer import DatasetChainer
from common.util.enums import EnvironmentType
from lime import lime_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import random

# python train.py --config="grapher_12_conv_gelu_config" --wandb-run-name="grapher only"

parser = argparse.ArgumentParser(
    description="Evaluation Config",
    add_help=True,
)

parser.add_argument(
    "--model-path",
    default=None,
    type=str,
    help="Give model path for evaluation.",
)
parser.add_argument(
    "--batch-size",
    default=1,
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
    "--images",
    type=int,
    default=4,
    help="Defines total images taken for sampling.",
)

parser.add_argument(
    "--features",
    type=int,
    default=10,
    help="Defines total features.",
)


def plot_explainability(
    images,
    total_images: int,
    model,
    dataset,
    features,
    n_classes: int,
    device: str,
    title: str = "Explainability",
):
    def model_lime(images):
        images = np.transpose(images, (0, 3, 1, 2))
        images = torch.from_numpy(images).to(device)
        output = model(images)
        return output.detach().cpu().numpy()
        # logger.info(model)

    def model_normal(images):
        images = torch.from_numpy(images).to(device)
        output = model(images)
        return output.detach().cpu().numpy()
        # logger.info(model)

    fig, ax = plt.subplots(total_images, 5)
    fig.suptitle(title)
    explainer = lime_image.LimeImageExplainer()

    for i in range(total_images):
        match = False
        while not match:
            image = random.choice(images)
            raw_image = Image.open(image.path)
            inputs, label = dataset.pre_process(image)
            output = model_normal(np.expand_dims(inputs, axis=0))
            match = np.argmax(output) == np.argmax(label)

        # Grad-Cam explainer
        target_layers = [model.backbone]
        input_tensor = torch.from_numpy(np.expand_dims(inputs, axis=0)).to(device)
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        label = np.argmax(label)
        targets = [ClassifierOutputTarget(label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(
            np.transpose(inputs, (1, 2, 0)), grayscale_cam, use_rgb=True
        )

        # Lime Explainer
        inputs = np.transpose(inputs, (1, 2, 0))
        explanation = explainer.explain_instance(
            inputs,
            model_lime,
            top_labels=5,
            hide_color=0,
            num_samples=1000,
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=features,
            hide_rest=False,
        )
        img_boundry1 = mark_boundaries(temp, mask)
        ax[i, 0].imshow(raw_image, cmap="gray")
        ax[i, 1].imshow(inputs)
        ax[i, 2].imshow(img_boundry1)
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=features,
            hide_rest=False,
        )
        img_boundry2 = mark_boundaries(temp, mask)
        ax[i, 3].imshow(img_boundry2)
        ax[i, 4].imshow(visualization)
    ax[0, 0].set_title("Original")
    ax[0, 1].set_title("Resized")
    ax[0, 2].set_title("Lime positive only")
    ax[0, 3].set_title("Lime all")
    ax[0, 4].set_title("Grad CAM")
    plt.show()


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
    device = cuda_info()
    dataset = get_dataset(
        args.dataset,
        environment=environment,
        augment_times=0,
        height=args.height,
        width=args.width,
    )
    train_files = dataset.get_train_files()
    test_files = dataset.get_test_files()

    config = get_config(
        "vig_pyramid_compact",
        "gelu",
        "conv",
        args.n_classes,
        2,
        args.height,
        args.width,
    )
    model = get_model(config)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    model.eval()
    # Training loop
    plot_explainability(
        train_files,
        args.images,
        model,
        dataset,
        args.features,
        args.n_classes,
        device,
        title="Train images",
    )
    plot_explainability(
        test_files,
        args.images,
        model,
        dataset,
        args.features,
        args.n_classes,
        device,
        title="Test images",
    )


if __name__ == "__main__":
    main()


x: int = 10

while x < 15:
    x += 1

print("Done:", x)
