import os
import shutil

from tqdm import tqdm

# root_dir = "./datasets/enhanced_mmcbnu/"
root_dir = "./datasets/VERA-fingervein/"

input_dir = os.path.join(root_dir, "cropped/bf")
output_dir = os.path.join(root_dir)

for subject in tqdm(os.listdir(input_dir)):
    subject_id = subject.split("-")[0]
    hands = ["L", "R"]
    for hand in hands:
        train_dir = os.path.join(output_dir, f"train/{subject_id}_{hand}")
        test_dir = os.path.join(output_dir, f"test/{subject_id}_{hand}")
        os.makedirs(train_dir)
        os.makedirs(test_dir)
        for i in range(1, 3):
            image_name = f"{subject_id}_{hand}_{i}"
            shutil.copyfile(
                os.path.join(input_dir, subject, image_name + ".png"), 
                os.path.join(train_dir if i == 1 else test_dir,image_name + ".png" ), 
            )
