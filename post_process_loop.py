import argparse
import json
from typing import Any, Dict, List, Tuple
from common.data_pipeline.dataset import get_dataset
from common.util.data_pipeline.dataset_chainer import DatasetChainer
import wandb
import torch
from train import get_config
from common.util.enums import EnvironmentType
from common.util.logger import logger
from common.train_pipeline.train import train
from common.evaluate_pipeline.evaluate import evaluate
import matlab
import matlab.engine

model_name = "vig_attention_at_last_pyramid_tiny"
printers = ["Canon", "DNP", "Digital"]
process_types = ["After", "Before"]
model_type = ["train", "test"]

parser = argparse.ArgumentParser(
    description="Model Loop Config",
    add_help=True,
)
parser.add_argument(
    "--train",
    default=False,
    type=bool,
    help="Train the model or just evaluate. by default it just evaluates.",
)
parser.add_argument(
    "--epochs",
    default=40,
    type=int,
    help="Number of epochs to train the model. By default ist 40.",
)


def main():
    args = parser.parse_args()
    act = "gelu"
    epochs = args.epochs
    pred_type = "conv"
    n_classes = 2
    height = 224
    width = 224
    batch_size = 192
    validate_after_epochs = 5
    learning_rate = 1e-5
    num_heads = 2
    augment_times = 19
    if args.train:
        for printer in printers:
            for process_type in process_types:
                wandb_run_name = f"{model_name}_{printer}_{process_type}"
                config = get_config(
                    model_name,
                    act,
                    pred_type,
                    n_classes,
                    num_heads,
                    height,
                    width,
                )

                if wandb_run_name:
                    wandb.init(
                        # set the wandb project where this run will be logged
                        project="morph-attack-detection",
                        name=wandb_run_name,
                        config={
                            "architecture": model_name,
                            "dataset": f"{printer}_{process_type}",
                            "epochs": epochs,
                            "activation": act,
                            "predictor_type": pred_type,
                        },
                    )
                try:
                    train(
                        config,
                        f"post_process_{printer}_{process_type}",
                        batch_size,
                        epochs,
                        EnvironmentType.PYTORCH,
                        wandb_run_name,
                        validate_after_epochs,
                        learning_rate,
                        None,
                        augment_times,
                        n_classes,
                        height,
                        width,
                    )
                except KeyboardInterrupt:
                    pass

                if wandb_run_name:
                    wandb.finish()
                    torch.cuda.empty_cache()

    print("Starting evaluation")

    eng = None
    try:
        eng = matlab.engine.start_matlab()
        script_dir = "/home/ubuntu/finger-vein-quality-assessement/EER"
        eng.addpath(script_dir)
    except Exception as e:
        logger.exception("Cannot initialise matlab engine", exc_info=e)

    all_datasets: Dict[str, Dict[str, Tuple[Any, Any, Any]]] = {}
    dataset_list: List[str] = []
    for printer in printers:
        all_datasets[printer] = {}
        for process_type in process_types:
            dataset_list.append(f"{printer}_{process_type}")
            print("Loading Dataset:", printer, process_type)
            all_datasets[printer][process_type] = DatasetChainer(
                datasets=[
                    get_dataset(
                        f"post_process_{printer}_{process_type}",
                        environment=EnvironmentType.PYTORCH,
                        augment_times=0,
                        height=height,
                        width=width,
                    )
                ],
            ).get_dataset(
                batch_size=batch_size,
                dataset_type=EnvironmentType.PYTORCH,
            )

    for model_t in model_type:
        for dprinter in printers:
            for dprocess_type in process_types:
                model = "models/checkpoints/best_"
                model += f"{model_t}_{model_name}_{dprinter}_{dprocess_type}.pt"
                wandb_run_name = model.split("/")[-1].split(".")[0]
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="morph-attack-detection",
                    name=wandb_run_name,
                    config={
                        "architecture": model_t,
                        "Trained On": f"{dprinter}_{dprocess_type}",
                    },
                )
                wandb.define_metric("evaluated_on")
                wandb.define_metric("accuracy", step_metric="evaluated_on")
                wandb.define_metric("recall", step_metric="evaluated_on")
                wandb.define_metric("precision", step_metric="evaluated_on")
                wandb.define_metric("eer", step_metric="evaluated_on")
                index = 0
                for printer in printers:
                    for process_type in process_types:
                        (train_ds, test_ds, validation_ds) = all_datasets[printer][
                            process_type
                        ]
                        try:
                            print("Model:", model)
                            results = evaluate(
                                (train_ds, test_ds, validation_ds),
                                model,
                                256,
                                EnvironmentType.PYTORCH,
                                n_classes,
                                height,
                                width,
                                eng=eng,
                            )
                            results["evaluated_on"] = index
                            wandb.log(results)
                            index += 1

                        except Exception as e:
                            print("Error while evaluating:", e)

                wandb.finish()

    print(*[f"{i}: {ds}" for i, ds in enumerate(dataset_list)])


if __name__ == "__main__":
    main()
