import torch.nn as nn
from tools.tools import batch_duplication
def loss_fn(outputs, atlas):
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
    pixel_loss = nn.MSELoss()(outputs[0], atlas)
    # differential_loss =
    loss = pixel_loss
    return loss

def MSE_loss(outputs, atlas):
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
    atlas = batch_duplication(atlas, outputs.shape[0])
    pixel_loss = nn.MSELoss().forward(outputs, atlas)
    # differential_loss =
    loss = pixel_loss
    return loss

def grad(vector_fields):
    """
    calculate the gradiant size of the vector fields
    Args:
        vector_fields:
            torch tensor of the form (batch_size, dims)
    Returns :
        torch tensor of size (batch_size, grad_size)
    """
    for d in vector_fields.shape[1:]:
        dim_slice = vector_fields[:,]