import nibabel as nib
import numpy as np
import os, glob
import torch
from torch.autograd import Variable
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


def network_input(data_dir, split_tet=(0.8, 0.1, 0.1), batch_size=1, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create a list of files from the folder
    glob_paths = glob.glob("")
    for ext in ('*.nii', '*.nii.gz', '.npz'):
        glob_paths.extend(glob.glob(os.path.join(data_dir, ext)))
    paths = list(glob_paths)

    generator = dataset_generator(paths, batch_size)
    while True:
        yield Variable(next(generator).to(device), requires_grad=True)


def ct_pet_data_generetor(folder_path, load_type, batch_size=1, load_labels=True, labels=[], data_type="ct"):
    """
    Data loader and generator for the dataset head neck pet ct
    :param data_type: "ct" or "pet"
    :param labels: list of labels to load. if empty and load_lables is true, will load every label
    :param load_labels (bool): if false, will not load any labels
    :param folder_path: path to the base folder containing ct, pet, struct folders
    :param load_type: "test" or "train"
    :return: yield
    """
    assert (load_type == "train" or load_type == "test")

    images_path = list(glob.glob(os.path.join(folder_path, data_type, "*.nii.gz")))
    test_list = ['HN-HGJ-022', 'HN-HGJ-074', 'HN-HGJ-091', 'HN-HGJ-001', 'HN-HGJ-002', 'HN-HGJ-009',
                 'HN-HGJ-010', 'HN-HGJ-012', 'HN-HGJ-013', 'HN-HGJ-016', 'HN-HGJ-018', 'HN-HGJ-019',
                 'HN-HGJ-020', 'HN-HGJ-028', 'HN-HGJ-029', 'HN-HGJ-035', 'HN-HGJ-036', 'HN-HGJ-040',
                 'HN-HGJ-042', 'HN-HGJ-044', 'HN-HGJ-045', 'HN-HGJ-050', 'HN-HGJ-051', 'HN-HGJ-052',
                 'HN-HGJ-053', 'HN-HGJ-059', 'HN-HGJ-062']
    train_list = [os.path.split(image_path)[-1].split(".")[0] for image_path in images_path]
    train_list = [image_name for image_name in train_list if image_name not in test_list]
    selected_data_names = train_list if load_type == "train" else test_list


    while True:
        random_indcies = np.random.randint(len(selected_data_names), size=batch_size)
        batch_data = np.array([])
        label_data_list = []

        for i in range(len(random_indcies)):
            sample_name =selected_data_names[random_indcies[i]]
            volume = load_file_for_stn(os.path.join(folder_path, data_type, sample_name + ".nii.gz"))
            if i == 0:
                batch_data = volume
            else:
                batch_data = np.append(batch_data, volume, axis=0)

            labels_paths = []
            labels_data = np.array([])
            if len(labels) == 0 and load_labels:
                labels_paths = list(glob.glob(os.path.join(folder_path, "struct",
                                                           sample_name, "*.nii.gz")))
            elif len(labels) > 0 and load_labels:
                for label in labels:
                    labels_paths.append(os.path.join(folder_path, "struct", sample_name, label + ".nii.gz"))
            for label_index in range(len(labels_paths)):
                label = load_file_for_stn(labels_paths[label_index])
                if label_index == 0:
                    labels_data = label
                else:
                    labels_data = np.append(labels_data, label, axis=0)

            label_data_list.append(labels_data)

        yield batch_data, label_data_list

if __name__ == "__main__":
    gen = ct_pet_data_generetor("/media/almog-lab/dataset/head-neck-ordered/", "train")
    data = next(gen)
    print(data)