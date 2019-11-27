import torch
import torch.functional as F
import scipy.ndimage.interpolation as sci
import numpy as np
import matplotlib.pyplot as plt

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

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)