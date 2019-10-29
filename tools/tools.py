import torch
import os
import time
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
    filename = time_sign + "_epoch-" + str(epoch)+".pt"
    path = os.path.join(save_dir, filename)
    return path


def roll(tensor, dim, shift=1, fill_pad=None):
    """
    numpy roll implementation. referenced from
    https://discuss.pytorch.org/t/implementation-of-function-like-numpy-roll/964/7

    Args:
        tensor:
        dim:
        shift:
        fill_pad:

    Returns:

    """
    if 0 == shift:
        return tensor

    elif shift < 0:
        shift = -shift
        gap = tensor.index_select(dim, torch.arange(shift))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=tensor.device)
        return torch.cat([tensor.index_select(dim, torch.arange(shift, tensor.size(dim))), gap], dim=dim)

    else:
        shift = tensor.size(dim) - shift
        gap = tensor.index_select(dim, torch.arange(shift, tensor.size(dim)))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=tensor.device)
        return torch.cat([gap, tensor.index_select(dim, torch.arange(shift))], dim=dim)


def shift_volume(tensor, shift=2, dim=0, constant=0):

    assert dim >= 0 & dim < 3

    if shift == 0:
        return tensor

    if shift>0:
        dim = dim *2 -2
    else:
        dim = dim *2 -1

    # 2 for starting padding, and 3 for dimension of the volume
    pad = [0]*2*3
    pad[dim] = shift
    raise NotImplementedError("torch didn't implement this when I wrote this :(")
    padded = F.pad(tensor, pad, constant)
    return padded

def shift_numpy_volume(array,shift,dim,constant=0):
    raise NotImplementedError()
if __name__ == "__main__":
    t = torch.arange(27).view(1,3,3,3,1)
    print(t)
    print(shift_volume(t))