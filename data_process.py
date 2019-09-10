import nibabel as nib
import numpy as np
import os, glob


def load_file(path):
    """
    Load a volume file to memory
    Args:
        path(string):
            path to the file, needs to be of '.nii', '.nii.gz' or '.npz' format only
    Returns:
        numpy array of the volume in float 64 format
    """

    if path.endswith(('.nii', '.nii.gz')):
        image = nib.load(path).get_data()

    elif path.endswith('.npz'):
        image = np.load(path)['arr_0']
    else:
        raise OSError("unknown file was loaded")
    return image.astype("float64")


def dataset_generator(paths, batch_size=1):


    while True:
        random_indcies = np.random.randint(len(paths), size=batch_size)
        batch_data = np.array([])


        for i in range(len(random_indcies)):


            volume = load_file(paths[random_indcies[i]])
            volume = volume[np.newaxis, np.newaxis, ...]

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