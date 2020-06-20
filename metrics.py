import torch.nn as nn
import torch
from tools.tools import batch_duplication, create_unit_grid, to_slice
import autoencoder.voxelmorph_losses as vl
import numpy as np
import matplotlib.pyplot as plt
def loss_mse_with_grad(outputs, atlas, grad_coef=0.001, pixel_coef=1000):
    """
    Calculate the Cross correlation loss, with added regularization of differentiability.
    from An Unsupervised Learning Model for Deformable Medical Image Registration
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Balakrishnan_An_Unsupervised_Learning_CVPR_2018_paper.pdf

    Args:
        outputs:
            numpy array of the [warp_images, warping_map]
        atlas:
            The atlas giving the to the network while training

    Returns (float):
        nn loss function
    """

    penalty = 'l1'
    volumes = outputs[0]
    batch_size = volumes.shape[0]
    vector_fields = outputs[1]
    # atlas = batch_duplication(atlas, batch_size)

    # calculate pixel loss using MSE
    pixel_loss = pixel_coef * nn.MSELoss().forward(volumes, atlas)

    # calucate the gradiants
    unit_grid = create_unit_grid(vector_fields.shape[1:-1]).to(vector_fields.device)
    gradiants = (unit_grid-vector_fields).abs().sum()# grad(vector_fields=vector_fields)

    if penalty == 'l1':
        grad_loss = grad_coef*gradiants.abs().sum()
    else:
        assert penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % penalty
        grad_loss = grad_coef*(gradiants*gradiants).sum()

    loss = pixel_loss + grad_loss

    print("pixel loss {}, grad loss {}, total loss {}".format(pixel_loss, grad_loss, loss))

    # normalize the loss by the batch size
    return loss / batch_size


def mask_regularization(mask, image_shape):
    relu = torch.nn.ReLU(True)
    mask = mask[0] # TODO batch fixes
    mask_size_loss = -(mask[1][0]-mask[0][0]) - (mask[1][1]-mask[0][1]) -(mask[1][2]-mask[0][2])

    mask_offscreen_loss = relu(-mask).sum() +\
                          relu(mask[:, 0] - image_shape[0]).sum() +\
                          relu(mask[:, 1] - image_shape[1]).sum() +\
                          relu(mask[:, 2] - image_shape[2]).sum()
    print(mask)
    return mask_size_loss + 10*mask_offscreen_loss

def pixel_loss_with_masking(warped, mask, atlas):
    slicing_square = to_slice(mask, atlas.shape[2:])
    return count_loss(warped[slicing_square],atlas[slicing_square] )


def MSE_loss(warped, atlas):
    """
    Calculate the Cross correlation loss, with added regularization of differentiability.
    from An Unsupervised Learning Model for Deformable Medical Image Registration
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Balakrishnan_An_Unsupervised_Learning_CVPR_2018_paper.pdf

    Args:
        outputs:
            numpy array of the [warp_images, warping_map]
        atlas:
            The atlas giving the to the network while training

    Returns (float):
        nn loss function
    """
    atlas = batch_duplication(atlas, warped.shape[0])
    pixel_loss = ((atlas[0]+3.024)*(warped - atlas[0])**2).mean()
    # differential_loss =
    loss = pixel_loss[None,...]
    return loss

def count_loss(warped, atlas):
    atlas = batch_duplication(atlas, warped.shape[0])
    pixel_loss = ((warped - atlas[0])**2).mean()
    # differential_loss =
    loss = pixel_loss[None,...]
    return loss

def vl_loss(outputs, atlas):
    volumes = outputs[0]
    batch_size = volumes.shape[0]
    vector_fields = outputs[1]
    # atlas = batch_duplication(atlas, batch_size)
    pixel_loss = vl.mse_loss(atlas, volumes)
    gradient_loss = vl.gradient_loss(vector_fields)

    return pixel_loss + gradient_loss
def grad(vector_fields):
    """
    calculate the gradient size of the vector fields
    Args:
        vector_fields:
            torch tensor of the form (batch_size, vol_shape, 3)
    Returns :
        torch tensor of size (batch_size, grad_size)
    """
    vol_shape = vector_fields.shape[1:-1]
    gradiants = torch.tensor([], requires_grad=True, device=vector_fields.device)

    for vector_field in vector_fields:
        # slice in every dim except batch dim and final dim (x,y,z)
        for i in range(len(vol_shape)):

            # slice that take the first element of the dim
            slicer1 = [slice(None)]*(len(vol_shape) + 1)
            slicer1[i] = slice(1, None, None)

            # slice that take the first element of the dim
            slicer2 = [slice(None)]*(len(vol_shape) + 1)
            slicer2[i] = slice(None, -1, None)

            # adding a zero element to fix dimension issues
            part_grad = vector_field[slicer1] - vector_field[slicer2]
            slicer1[i] = slice(1, 2, None)
            zero = torch.zeros_like(vector_field[slicer1])
            part_grad = torch.cat((zero, part_grad),i)
            # part_grad
            # permute dimensions to put the ith dimension first
            gradiants = torch.cat((gradiants,part_grad))
    return gradiants

def dice_loss(y_true, y_pred):
    top = 2 * (y_true * y_pred).sum()
    bottom = (y_true + y_pred).sum()
    dice = torch.mean(top / bottom)
    return -dice

if __name__ == "__main__":
    loss_ncc(0,0)