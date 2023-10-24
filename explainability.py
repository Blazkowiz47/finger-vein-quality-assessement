"""
Main explainability file.
calls the evaluate pipeline with configs.
"""
import argparse
from PIL import Image
import torch
from common.util.logger import logger
from common.util.enums import EnvironmentType

from common.train_pipeline.model.model import get_model
from tqdm import tqdm
import numpy as np
from common.data_pipeline.dataset import get_dataset

from train import get_config
from common.util.data_pipeline.dataset_chainer import DatasetChainer
from common.util.enums import EnvironmentType
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

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
    device = "cpu"
    datasets = [args.dataset]
    train_dataset, test_dataset, validation_dataset = DatasetChainer(
        datasets=[
            get_dataset(
                dataset,
                environment=environment,
                augment_times=0,
                height=args.height,
                width=args.width,
            )
            for dataset in datasets
        ],
    ).get_dataset(
        batch_size=batch_size,
        dataset_type=environment,
    )

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

    # logger.info(model)
    def model_mod(inputs):
        inputs = np.transpose(inputs, (0, 3, 1, 2))
        return model(torch.from_numpy(inputs))

    # Training loop
    with torch.no_grad():
        for index, dataset in enumerate(
            [train_dataset, test_dataset, validation_dataset]
        ):
            if not dataset:
                continue
            i = 0
            for inputs, labels in tqdm(dataset, desc="Train:"):
                if inputs.shape[0] == 1:
                    inputs = torch.cat((inputs, inputs), 0)  # pylint: disable=E1101
                    labels = torch.cat((labels, labels), 0)  # pylint: disable=E1101
                inputs = inputs.to(device).numpy().astype(np.float32)
                labels = labels.to(device).numpy().astype(np.float32)
                inputs = np.transpose(inputs, (0, 2, 3, 1))
                explainer = lime_image.LimeImageExplainer()
                explanation = explainer.explain_instance(
                    inputs[0],
                    model_mod,
                    top_labels=5,
                    hide_color=0,
                    num_samples=1000,
                )
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0],
                    positive_only=True,
                    num_features=5,
                    hide_rest=False,
                )
                img_boundry1 = mark_boundaries(temp / 255.0, mask)
                image = np.array(img_boundry1) * 255
                image = image.astype(np.uint8)
                im = Image.fromarray(image)
                im.save(f"{index}_{i}.jpg")
                image = np.array(inputs[0]) * 255
                image = image.astype(np.uint8)
                im = Image.fromarray(image)
                im.save(f"input_{index}_{i}.jpg")
                if i == 4:
                    break
                i += 1


if __name__ == "__main__":
    main()
