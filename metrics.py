import torch.nn as nn
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
    pixel_loss = nn.MSELoss()(outputs[0], atlas)
    # differential_loss =
    loss = pixel_loss
    return loss