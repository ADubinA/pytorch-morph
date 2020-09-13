from scipy.interpolate import RegularGridInterpolator
import numpy as np
import open3d as o3d
from emd_module import *
import torch
def volume_to_pointcloud(volume: np.ndarray, sample_num, intensity_range=None):
    """
    generates a pointcloud from a the given volume.
    Args:
        volume: numpy array of dim 3. samples will be taken from this.
        sample_num: number of sample to be taken from the volume
        intensity_range: tuple of the form (low, high) or None. samples will be within that value.
            if None, pointcloud will be from any range.

    Returns: numpy array of the form

    """
    # get all good values
    if not intensity_range:
        intensity_range = (-float("infinity"), float("infinity"))
    locations = np.argwhere((volume > intensity_range[0]) & (volume <= intensity_range[1]))
    colors = volume[np.where((volume > intensity_range[0]) & (volume <= intensity_range[1]))]
    colors = np.tile(colors.reshape(-1,1),(1,3))
    colors += 1025
    colors /= 3000
    assert colors.shape[0]>=sample_num,\
        "wanted {} samples, but the filtered image has only {} values".format(sample_num,colors.shape[0])

    # get only a few of them
    idx = np.random.randint(colors.shape[0], size=sample_num)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(locations[idx])
    pcd.colors = o3d.utility.Vector3dVector(colors[idx])
    # o3d.visualization.draw_geometries([pcd])

    # return pcd
    return locations[idx].reshape(1,sample_num,3)
    # # generating interpolation function
    # x = np.linspace(0, volume.shape[0], 1)
    # y = np.linspace(0, volume.shape[1], 1)
    # z = np.linspace(0, volume.shape[2], 1)
    # interpolating_function = RegularGridInterpolator((x, y, z), volume)

# def find_best_z_iterative(image,ref,stride=20):
#     for i in range(0,ref.shape[3])

if "__main__" == __name__:
    from data_process import *
    atlas = load_file(r"F:\dataset\atlas-f-arms-down.nii.gz")

    # atlas = atlas.cpu().detach().numpy()[0,0,:,:,:]
    atlas_pcd1 = torch.from_numpy(volume_to_pointcloud(atlas[:,:,100:150],8192,(300,700 )))
    atlas_pcd2 = torch.from_numpy(volume_to_pointcloud(atlas[:,:,30:80],8192,(300,700 )))

    # o3d.visualization.draw_geometries([atlas_pcd])
    emd_m = emdModule().cpu()
    dis, ass = emd_m(atlas_pcd1, atlas_pcd2, 0.05, 3000)
    print(np.sqrt(dis.cpu()).mean())