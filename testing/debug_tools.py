import torch
import torch.functional as F
import scipy.ndimage.interpolation as sci
import numpy as np
def rotate_vol(tensor,angle):
    """

    :param tensor: of the form (batch_size,1,(volume))
    :param angle:
    :return:
    """
    rotated = sci.rotate(tensor[0, 0, :, :, :], angle, reshape=False, mode="nearest")
    t =  torch.tensor(rotated[None, None, :, :, :], requires_grad=True)
    return t

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


def shift_volume(tensor, dim=0, shift=2, constant=0):

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


def print_back(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                print_back(n[0])

def set_zero_except_layer(volumes, layer, dim, constant=0):

    # dim expect (batch, 1, volume_dim)
    dim = dim+2
    slicer = [slice(None)] * 5

    slicer[dim] = slice(0, layer)
    volumes[slicer] = constant

    slicer[dim] = slice(layer+1, -1)
    volumes[slicer] = constant

    return volumes

