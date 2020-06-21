import torch
import os
import time
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plt_show_slice(tensor_image, slice_index = 15):
    plt.figure()
    plt.imshow(tensor_image.detach().cpu().numpy()[0, 0, :, :, slice_index])

def slice_5d(min_slice, max_slice, image_size):
    out_slice = []
    out_slice.append(slice(None))
    out_slice.append(slice(None))
    out_slice.append(slice(max(min_slice[0], 0), min(max_slice[0], image_size[0])))
    out_slice.append(slice(max(min_slice[1], 0), min(max_slice[1], image_size[1])))
    out_slice.append(slice(max(min_slice[2], 0), min(max_slice[2], image_size[2])))
    return out_slice


def to_slice(x, image_size):
    # TODO handle out of bounds?
    # Todo handle int?
    x = x[0] # TODO batch fixes
    x = x.detach().cpu().numpy()
    x[0] = np.floor(x[0])
    x[1] = np.ceil(x[1])
    x = x.astype(np.int)
    return slice_5d(x[0],x[1], image_size)

def random_image_slice(image, min_slice, max_slice, slice_size=np.array([40, 40, 20]) , default_color=None):
    default_color = default_color or image[0,0,0,0,0]
    low_random_slice = np.array([20,20,5])
    # low_random_slice = np.random.randint(min_slice, max_slice, size=3)
    high_random_slice = low_random_slice+slice_size
    image_slice = slice_5d(low_random_slice, high_random_slice, image.shape[2:])
    sliced_image = torch.zeros_like(image) + default_color
    sliced_image[image_slice] = image[image_slice]
    return sliced_image



def batch_duplication(tensor, batch_size):
    # tensor = tensor.unsqueeze(0)
    return torch.stack(list(tensor for _ in range(batch_size)))

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
    return path + ".png"


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