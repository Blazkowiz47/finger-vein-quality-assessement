import json
import wandb
import torch
from train import get_config
from common.util.enums import EnvironmentType
from common.train_pipeline.train import train
from common.evaluate_pipeline.evaluate import evaluate

model_name = "vig_attention_pyramid_tiny"
dataset_list = ["lma", "mipgan_1", "mipgan_2", "stylegan_iwbf"]
model_type = ["train", "test"]
all_results = {}
def main(train_models:bool = False):
    if train_models:
        for dataset in dataset_list:
            wandb_run_name = f"{model_name}_{dataset}"
            act = 'gelu'
            pred_type = 'conv'
            epochs = 25 
            n_classes = 2
            height = 224
            width = 224
            batch_size = 192 
            validate_after_epochs = 5
            learning_rate = 1e-4
            num_heads = 2
            augment_times = 19
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
                        "activation":act ,
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
                    all_results[model_name]["best" + model_t][dataset_model][dataset] = evaluate(
                            ["dnp_"+dataset],
                            model,
                            512,
                            EnvironmentType.PYTORCH,
                            n_classes,
                            height,
                            width,
                            )
                except:
                    ...
    with open(f"results/{model_name}.json", "w+") as fp:
        json.dump(all_results, fp) 
main()
