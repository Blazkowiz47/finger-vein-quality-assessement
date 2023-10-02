import argparse
import json
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
all_results = {}

def prettier(data):
    """
    Converts into better json format
    """
    results = {}
    metrics = ["accuracy", "precision", "recall", "eer"]
    dataset_type = ["train", "test"]
    model_types = ["besttrain", "besttest"]
    model_data = data
    for metric in metrics:
        results[metric] = {}
        for tprinter in printers:
            results[metric][tprinter] = {}
            for tprocess_type in process_types:
                results[metric][tprinter][tprocess_type]  = {}
                for model_type in model_types:
                    results[metric][tprinter][tprocess_type][model_type] = {}
                    for split_type in dataset_type:
                        results[metric][tprinter][tprocess_type][model_type][split_type] = {}
                        for eprinter in printers:
                            results[metric][tprinter][tprocess_type][model_type][split_type][eprinter] = {}
                            for eprocess_type in process_types:
                                   results[metric][tprinter][tprocess_type][model_type][split_type][eprinter][eprocess_type]  = round(
                                       model_data[model_type][tprinter][tprocess_type][eprinter][eprocess_type][
                                           split_type
                                           ][metric]
                                       * 100,
                                       3,
                                       )
    return results



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
def main():
    args = parser.parse_args()
    act = "gelu"
    epochs = 40 
    pred_type = "conv"
    n_classes = 2
    height = 224
    width = 224
    batch_size = 192
    validate_after_epochs = 5
    learning_rate = 1e-4
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
                        project="finger-vein-recognition",
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

    all_results = {}
    eng = None 
    try: 
        eng = matlab.engine.start_matlab()
        script_dir = f"/home/ubuntu/finger-vein-quality-assessement/EER"
        eng.addpath(script_dir)
    except:
        logger.exception("Cannot initialise matlab engine")

    for printer in printers:
        for process_type in process_types:
            print("Dataset:", printer, process_type)
            train_dataset, test_dataset, validation_dataset = DatasetChainer(
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
                if not all_results.get("best" + model_t):
                    all_results["best" + model_t] = {}
                for dprinter in printers:
                    for dprocess_type in process_types:
                        model = f"models/checkpoints/best_{model_t}_{model_name}_{dprinter}_{dprocess_type}.pt"
                        if not all_results["best" + model_t].get(dprinter):
                            all_results["best" + model_t][dprinter] = {}
                        if not all_results["best" + model_t][dprinter].get(dprocess_type):
                            all_results["best" + model_t][dprinter][dprocess_type] = {}
                        try:
                            print("Model:", model)
                            all_results["best" + model_t][dprinter][dprocess_type] = evaluate(
                                    (train_dataset, test_dataset, validation_dataset),
                                    model,
                                    512,
                                    EnvironmentType.PYTORCH,
                                    n_classes,
                                    height,
                                    width,
                                    eng=eng,
                                    )
                        except Exception as e:
                            print("Error while evaluating:", e)


    formatted_results = prettier(all_results)
    with open(f"results/{model_name}_post_process.json", "w+") as fp:
        json.dump(formatted_results, fp)

if __name__ == '__main__':
    main()