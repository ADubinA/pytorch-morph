import torch
import os
import time
import numpy as np
import torch.nn.functional as F

def batch_duplication(tensor, batch_size):
    # tensor = tensor.unsqueeze(0)
    return torch.cat(list(tensor for _ in range(batch_size)))

def set_path(path, is_folder=True):
    """
    create folders if not already exist
    Args:
        path(string):
            path for the file or folder
        is_folder(bool):
            set to True if the path is for a folder, and false if it is for a file
    Returns:
        None
    """
    if not is_folder:
        path = os.path.dirname(path)
    if not os.path.isdir(path):
        os.mkdir(path)

def save_model_string(save_dir, epoch):
    """
    returns a string for the model save name.
    Args:
        epoch(int):
            epoch number
        save_dir(str):
            save directory for the file
    Returns(string):
        the proper file name with prefix
    """
    path = save_string(save_dir, epoch)
    return path + ".pt"


def save_sample_string(save_dir, epoch):
    path = save_string(save_dir, epoch)
    return path + ".jpg"



def save_string(save_dir, epoch):
    time_sign = time.strftime("%Y%m%d-%H%M%S")
    filename = time_sign + "_epoch-" + str(epoch)
    path = os.path.join(save_dir, filename)
    return path


def create_unit_grid(shape):
    # make sure that the shape is not a weird torch.Shape
    if type(shape) is torch.Size:
        shape = [int(s) for s in shape]

    # set the final shape and the new tensor (dimx,dimy,..., num_of_dims)
    new_shape = list(shape)
    new_shape.insert(0, len(new_shape))
    new_shape = tuple(new_shape)
    tensor = torch.zeros(new_shape)

    shifted_dim = [2,1,0]

    for dim in range(len(shape)):
        a = np.arange(-1, stop=1, step=2 / shape[dim])

        # tile on the other dimensions
        lshape = list(shape)
        lshape.pop(dim)
        lshape.append(1)  # this means that var a should be tiled once on it's dimension
        a = np.tile(a, tuple(lshape))

        # transpose to the right shape
        dims = list(range(len(shape)))
        dims.pop(-1)
        dims.insert(dim, len(shape)-1)
        a = a.transpose(tuple(dims))
        tensor[shifted_dim[dim]] = torch.tensor(a)

    # permute = list(range(len(new_shape)))
    # permute.insert(len(new_shape),permute.pop(0))
    # tensor = tensor.permute(tuple(permute))
    tensor = tensor.permute((1,2,3,0))
    tensor = tensor[None, :, :, :, :]
    return tensor


if __name__ == "__main__":
    pass