import os
import shutil

from tqdm import tqdm

input_dir = (
    "/home/blazkowiz47/work/finger-vein-quality-assessement/datasets/MMCBNU_6000/ROIs"
)
output_dir = (
    "/home/blazkowiz47/work/finger-vein-quality-assessement/datasets/MMCBNU_6000/oROIs"
)

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
