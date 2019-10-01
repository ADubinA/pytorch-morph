import torch
import os
import time


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

def save_string(save_dir, epoch):
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
    t =  time.strftime("%Y%m%d-%H%M%S")
    filename = t + "_epoch-" + str(epoch)+".pt"
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


if __name__ == "__main__":
    t = torch.arange(27*2).view(2,3,3,3)
    print(t)
    print(roll(t,dim=1))