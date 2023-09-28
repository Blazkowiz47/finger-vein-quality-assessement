import os
import numpy as np
from scipy.io import loadmat, savemat

def convert():
    files = os.listdir("results")
    for file in files:
        if not file.endswith("mat"):
            continue
        name = file.split('.')[0]
        matrix = loadmat("results/"+file)
        data = matrix['data']
        matrix = np.array(data)
        genuine = data[data[:,1] == 1.0]
        imposter = data[data[:,0] == 1.0 ] 
        savemat("results/"+name+"_split.mat", {"genuine": genuine[:,3], "morphed": imposter[:, 3]})

if __name__ == "__main__":
    convert()
