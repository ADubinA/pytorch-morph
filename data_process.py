import nibabel as nib
import numpy as np

def load_file(path):
    """
    Load a volume file to memory
    Args:
        datafile:

    Returns:
        numpy array of the volume in float 64 format
    """

    if path.endswith(('.nii', '.nii.gz', '.mgz')):
        image = nib.load(path).get_data()

    elif path.endswith('.npz'):
        image = np.load(path)['arr_0']
    else:
        raise OSError("unknown file was loaded")
    return image.astype("float64")


def dataset_generator(paths, batch_size=1):


    while True:
        random_indcies = np.random.randint(len(paths), size=batch_size)
        batch_data = []

        for i in random_indcies:
            volume = load_file(paths[i])
            volume = volume[np.newaxis, ..., np.newaxis]
            batch_data.append(volume)

        if batch_size > 1:
            return_vals = [np.concatenate(batch_data, 0)]
        else:
            return_vals = [batch_data[0]]

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
        yield tuple(return_vals)