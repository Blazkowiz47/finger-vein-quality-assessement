"""
Trains everything
"""
from multiprocessing import Pool
from typing import Any, List, Optional
import numpy as np
import torch
from torch.nn import CosineSimilarity
from torch.nn.functional import adaptive_avg_pool2d
from torch.optim import lr_scheduler as lr_scheduler
from tqdm import tqdm

# from common.data_pipeline.mmcbnu.dataset import DatasetLoader as mmcbnu
from common.train_pipeline.config import ModelConfig

# from common.data_pipeline.fvusm.dataset import DatasetLoader as fvusm
from common.data_pipeline.dataset import get_dataset

from common.train_pipeline.model.model import get_model
from common.util.logger import logger
from common.util.data_pipeline.dataset_chainer import DatasetChainer
from common.util.enums import EnvironmentType
import matlab
import matlab.engine

# To watch nvidia-smi continuously after every 2 seconds: watch -n 2 nvidia-smi


def cuda_info() -> str:
    """
    Prints cuda info.
    """

    device = torch.device(  # pylint: disable=E1101
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # pylint: disable=E1101
    logger.info("Using device: %s", device)

    # Additional Info when using cuda
    if device.type == "cuda":
        logger.info(torch.cuda.get_device_name(0))
        logger.info("Memory Usage:")
        logger.info(
            "Allocated: %s GB",
            round(torch.cuda.memory_allocated(0) / 1024**3, 1),
        )
        logger.info(
            "Cached: %s GB", round(torch.cuda.memory_reserved(0) / 1024**3, 1)
        )
    return device


def train(
    config: ModelConfig,
    dataset: str,
    batch_size: int = 10,
    epochs: int = 1,
    environment: EnvironmentType = EnvironmentType.PYTORCH,
    log_on_wandb: Optional[str] = None,
    validate_after_epochs: int = 5,
    learning_rate: float = 1e-3,
    continue_model: Optional[str] = None,
    augment_times: int = 0,
    n_classes: int = 301,
    height: int = 60,
    width: int = 120,
    pretrained_model_path: Optional[str] = None,
    pretrained_predictor_classes: Optional[int] = None,
    eng: Any = None,
):
    """
    Contains the training loop.
    """

    try:
        eng = matlab.engine.start_matlab()
        script_dir = "/home/ubuntu/finger-vein-quality-assessement/EER"
        eng.addpath(script_dir)
    except Exception:
        logger.exception("Cannot initialise matlab engine")

    device = cuda_info()

    if continue_model:
        model = get_model(config)
        model.load_state_dict(torch.load(continue_model))
        model = model.to(device)
    else:
        model = get_model(
            config,
            pretrained_model_path=pretrained_model_path,
            pretrained_predictor_classes=pretrained_predictor_classes,
        ).to(device)
    logger.info(model)
    logger.info("Total parameters: %s", sum(p.numel() for p in model.parameters()))
    logger.info(
        "Total trainable parameters: %s",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    train_dataset, validation_dataset, _ = DatasetChainer(
        datasets=[
            get_dataset(
                dataset,
                environment=environment,
                augment_times=0,
                height=height,
                width=width,
            ),
        ],
    ).get_dataset(
        batch_size=batch_size,
        dataset_type=environment,
    )

    _ = cuda_info()
    enroll_x = []
    enroll_y = []
    probe_x = []
    probe_y = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(train_dataset, desc="Compiling the features"):
            inputs = inputs.cuda().float()
            inputs = model.stem(inputs)
            inputs = model.backbone(inputs)
            inputs = adaptive_avg_pool2d(inputs, 1)
            inputs = inputs.squeeze()
            enroll_x.append(inputs.detach().cpu())
            enroll_y.append(labels)

        for inputs, labels in tqdm(validation_dataset, desc="Validation:"):
            inputs = inputs.cuda().float()
            inputs = model.stem(inputs)
            inputs = model.backbone(inputs)
            inputs = adaptive_avg_pool2d(inputs, 1)
            inputs = inputs.squeeze()
            probe_x.append(inputs.detach().cpu())
            probe_y.append(labels)

        enroll_x = torch.cat(enroll_x)
        enroll_y = torch.cat(enroll_y)
        probe_x = torch.cat(probe_x)
        probe_y = torch.cat(probe_y)
        ### Do the genuine score comparisons here
        cosineg: List[float] = []
        cosinem: List[float] = []
        genuine: List[float] = []
        morphed: List[float] = []

        def get_distances(x, y):
            genuine_euclidean, genuine_cosine, morphed_euclidean, morphed_cosine = (
                [],
                [],
                [],
                [],
            )
            for p, py in zip(probe_x, probe_y):
                if torch.argmax(y) == torch.argmax(py):
                    # Genuine
                    genuine_euclidean.append((x - p).square().sqrt().sum().item())
                    genuine_cosine.append(CosineSimilarity(0)(x, p))
                else:
                    # Imposter
                    morphed_euclidean.append((x - p).square().sqrt().sum().item())
                    morphed_cosine.append(CosineSimilarity(0)(x, p))
            return genuine_euclidean, morphed_euclidean, genuine_cosine, morphed_cosine

        with Pool(24) as p:
            result = p.map(get_distances, zip(enroll_x, enroll_y))
            for ge, me, gc, mc in result:
                genuine.extend(ge)
                morphed.extend(me)
                cosineg.extend(gc)
                cosinem.extend(mc)

        genuine = np.array(genuine)
        genuine = (genuine - genuine.min()) / (genuine.max() - genuine.min())
        morphed = np.array(morphed)
        morphed = (morphed - morphed.min()) / (morphed.max() - morphed.min())
        morphed = 1 - morphed
        eer, far, ffr = eng.EER_DET_Spoof_Far(
            genuine, morphed, matlab.double(10000), nargout=3
        )
        far = np.array(far)
        ffr = np.array(ffr)
        one = np.argmin(np.abs(far - 1))
        pointone = np.argmin(np.abs(far - 0.1))
        pointzeroone = np.argmin(np.abs(far - 0.01))
        print("For euclidean distance")
        print("EER:", eer)
        print("TAR 1%:", one)
        print("TAR 0.1%:", pointone)
        print("TAR 0.01%:", pointzeroone)

        genuine = np.array(cosineg)
        genuine = (genuine - genuine.min()) / (genuine.max() - genuine.min())
        morphed = np.array(cosinem)
        morphed = (morphed - morphed.min()) / (morphed.max() - morphed.min())
        morphed = 1 - morphed
        eer, far, ffr = eng.EER_DET_Spoof_Far(
            genuine, morphed, matlab.double(10000), nargout=3
        )
        far = np.array(far)
        ffr = np.array(ffr)
        one = np.argmin(np.abs(far - 1))
        pointone = np.argmin(np.abs(far - 0.1))
        pointzeroone = np.argmin(np.abs(far - 0.01))
        print("For cosine similarity")
        print("EER:", eer)
        print("TAR 1%:", one)
        print("TAR 0.1%:", pointone)
        print("TAR 0.01%:", pointzeroone)
