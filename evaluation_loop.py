from common.util.enums import EnvironmentType
from common.evaluate_pipeline.evaluate import evaluate

dataset_list = ["lma", "mipgan_1", "mipgan_2", "stylegan_iwbf"]
model_type = ["train", "test"]
def main():
    for dataset_model in dataset_list:
        for model in model_type:
            model = f"models/checkpoints/best_{model}_vig_pyramid_{dataset_model }.pt"
            for dataset in dataset_list:
                try:
                    print("Model:", model)
                    print("Dataset:", dataset)
                    evaluate(
                            ["dnp_"+dataset],
                            model,
                            512,
                            EnvironmentType.PYTORCH,
                            2,
                            224,
                            224,
                            )
                except:
                    ...

main()
