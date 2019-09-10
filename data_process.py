import nibabel as nib
import numpy as np
import os, glob
import torch
import pydicom
def load_file(path, dict_key="arr_0"):
    """
    Load a volume file to memory
    Args:
        path(string):
            path to the file, needs to be of '.nii', '.nii.gz' or '.npz' format only
        dict_key(string):
            optional key for numpy type file
    Returns:
        numpy array of the volume in float 64 format
    """

    if path.endswith(('.nii', '.nii.gz')):
        image = nib.load(path).get_data()

    elif path.endswith('.npz'):
        image = np.load(path)[dict_key]
    elif path.endswith(".dcm"):
        image = pydicom.read_file(path)
        image = image.pixel_array

    else:
        raise OSError("unknown file was loaded")
    return image.astype("float64")


def load_file_for_stn(path):
    """
    Load a volume file to memory to be used by the STN
    Args:
        path(string):
            path to the file, needs to be of '.nii', '.nii.gz' or '.npz' format only
    Returns:
        numpy array of the volume in float 64 format
    """
    numpy_file = load_file(path)[np.newaxis, np.newaxis, ...]
    return torch.from_numpy(numpy_file).float()


def dataset_generator(paths, batch_size=1):


    while True:
        random_indcies = np.random.randint(len(paths), size=batch_size)
        batch_data = np.array([])


        for i in range(len(random_indcies)):


            volume = load_file_for_stn(paths[random_indcies[i]])
            if i == 0:
                batch_data = volume
            else:
                batch_data = np.append(batch_data, volume, axis=0)




        # # also return segmentations
        # if return_segs:
        #     X_data = []
        #     for idx in idxes:
        #         X_seg = load_volfile(vol_names[idx].replace('norm', 'aseg'))
        #         X_seg = X_seg[np.newaxis, ..., np.newaxis]
        #         X_data.append(X_seg)
        #
        #     if batch_size > 1:
        #         return_vals.append(np.concatenate(X_data, 0))
        #     else:
        #         return_vals.append(X_data[0])
        yield batch_data


def network_input(data_dir, split_tet=(0.8, 0.1, 0.1), batch_size=1):

    # create a list of files from the folder
    glob_paths = glob.glob("")
    for ext in ('*.nii', '*.nii.gz', '.npz'):
        glob_paths.extend(glob.glob(os.path.join(data_dir, ext)))
    paths = list(glob_paths)

    generator = dataset_generator(paths, batch_size)
    while True:
       yield next(generator)