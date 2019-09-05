import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.ndimage import zoom
path = r"D:\LIDC-IDRI_npz"
newpath =  r"D:\LIDC-IDRI_npz_small"
for file in glob.glob(os.path.join(path, "*.npz")):
    with np.load(file) as data:
        image = data["arr_0"]
        new_array = zoom(image, (0.25, 0.25, 0.25))
        np.savez(os.path.join(newpath,os.path.split(file)[-1]),new_array)
        print(os.path.join(newpath,os.path.split(file)[-1]))
