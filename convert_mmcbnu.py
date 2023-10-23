import os
import shutil

from tqdm import tqdm

root_dir = "./datasets/enhanced_mmcbnu/"
# root_dir = "./datasets/MMCBNU_6000/"

input_dir = os.path.join(root_dir, "ROIs")
output_dir = os.path.join(root_dir, "oROIs")

for subject in tqdm(os.listdir(input_dir)):
    for finger in os.listdir(os.path.join(input_dir, subject)):
        odir = os.path.join(output_dir, subject + "_" + finger)
        os.makedirs(odir)
        images = [
            x
            for x in os.listdir(os.path.join(input_dir, subject, finger))
            if x.endswith("bmp")
        ]
        for image in images:
            shutil.copyfile(
                os.path.join(input_dir, subject, finger, image),
                os.path.join(odir, image),
            )

for class_id in tqdm(os.listdir(output_dir)):
    images = [
        x
        for x in os.listdir(os.path.join(output_dir, class_id))
        if int(x.split(".")[0]) > 7
    ]
    odir = os.path.join(root_dir, "test", class_id)
    os.makedirs(odir)
    for image in images:
        os.replace(os.path.join(output_dir, class_id, image), os.path.join(odir, image))

os.replace(output_dir, os.path.join(root_dir, "train"))
