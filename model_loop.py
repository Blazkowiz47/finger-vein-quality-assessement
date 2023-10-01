import argparse
import json
import wandb
import torch
from train import get_config
from common.util.enums import EnvironmentType
from common.train_pipeline.train import train
from common.evaluate_pipeline.evaluate import evaluate


def prettier(data):
    """
    Converts into better json format
    """
    results = {}
    metrics = ["accuracy", "precision", "recall"]
    dataset_type = ["train", "test"]
    model_types = ["besttrain", "besttest"]
    datasets = ["lma", "mipgan_1", "mipgan_2", "stylegan_iwbf"]

    model_data = data[model_name]
    for metric in metrics:
        results[metric] = {}
        for trained_on in datasets:
            results[metric][trained_on] = {}
            for model_type in model_types:
                results[metric][trained_on][model_type] = {}
                for split_type in dataset_type:
                    results[metric][trained_on][model_type][split_type] = {}
                    for evaluated_on in datasets:
                        results[metric][trained_on][model_type][split_type][
                            evaluated_on
                        ] = round(
                            model_data[model_type][trained_on][evaluated_on][
                                split_type
                            ][metric]
                            * 100,
                            3,
                        )
    return results


model_name = "vig_attention_at_last_pyramid_tiny"
dataset_list = ["lma", "mipgan_1", "mipgan_2", "stylegan_iwbf"]
model_type = ["train", "test"]
all_results = {}

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
        for dataset in dataset_list:
            wandb_run_name = f"{model_name}_{dataset}"
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
                        "dataset": dataset,
                        "epochs": epochs,
                        "activation": act,
                        "predictor_type": pred_type,
                    },
                )
            try:
                train(
                    config,
                    "dnp_" + dataset,
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

    all_results[model_name] = {}
    for model_t in model_type:
        all_results[model_name]["best" + model_t] = {}
        for dataset_model in dataset_list:
            model = f"models/checkpoints/best_{model_t}_{model_name}_{dataset_model}.pt"
            all_results[model_name]["best" + model_t][dataset_model] = {}
            for dataset in dataset_list:
                try:
                    print("Model:", model)
                    print("Dataset:", dataset)
                    all_results[model_name]["best" + model_t][dataset_model][
                            dataset
                            ] = evaluate(
                                    ["dnp_" + dataset],
                                    model,
                                    512,
                                    EnvironmentType.PYTORCH,
                                    n_classes,
                                    height,
                                    width,
                                    )
                    for split_type in model_type:
                        try:
                            import matlab
                            import matlab.engine
                            from scipy.io import load_mat
                            mat_file_path = f"best_{model_t}_{model_name}_{dataset_model}_{split_type}_dnp_{dataset}.mat"
                            eng = matlab.engine.start_matlab()
                            content = load_mat(mat_file_path)
                            script_dir = f"./EER"
                            eng.addpath(script_dir)
                            genuine = matlab.double(content['genuine'])
                            morphed  = matlab.double(content['morphed'])
                            eer, far, ffr = eng.EER_DET_Spoof_far(genuine,morphed , 10000, nargout=3)
                            all_results[model_name]["best" + model_t][dataset_model][dataset][split_type]['eer'] = eer 
                        except Exception as e:
                            print(e)


                except:
                    print("Error while evaluating")
                

    formatted_results = prettier(all_results)
    with open(f"results/{model_name}.json", "w+") as fp:
        json.dump(formatted_results, fp)

if __name__ == '__main__':
    main()
