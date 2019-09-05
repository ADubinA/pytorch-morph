import numpy as np
import matplotlib.pyplot as plt
import os
import glob
path = r"D:\LIDC-IDRI_npz"
# for file in glob.glob(os.path.join(path, "*.npz")):
#     with np.load(file) as data:
#         print(data["arr_0"].shape)
f, axarr = plt.subplots(2,2)

file1 = os.path.join(path,"5.npz")
with np.load(file1) as data:
        axarr[0,0].imshow(data["arr_0"][0])
        axarr[0, 1].imshow(data["arr_0"][95])
file1 = os.path.join(path,"40.npz")
with np.load(file1) as data:
        axarr[1,0].imshow(data["arr_0"][0])
        axarr[1, 1].imshow(data["arr_0"][95])
plt.show()