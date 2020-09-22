from scipy.interpolate import RegularGridInterpolator
import numpy as np
import open3d as o3d
# from emd_module import *
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import interpolation
import tqdm

def emd_scipy(pcl1,pcl2):
    d = cdist(pcl1, pcl2,'euclidean')
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / min(pcl1.shape[0], pcl2.shape[0])


def volume_to_pointcloud(volume: np.ndarray, sample_num,color=None, intensity_range=None):
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

    if locations.shape[0] <= sample_num:
        print("wanted {} samples, but the filtered image has only {} values".format(sample_num,colors.shape[0]))
        return False
    # get only a few of them
    idx = np.random.randint(colors.shape[0], size=sample_num)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(locations[idx])
    if color is None:
        pcd.colors = o3d.utility.Vector3dVector(colors[idx])
    else:
        pcd.paint_uniform_color(color)
    # o3d.visualization.draw_geometries([pcd])

    return pcd
    # return locations[idx]
    # # generating interpolation function
    # x = np.linspace(0, volume.shape[0], 1)
    # y = np.linspace(0, volume.shape[1], 1)
    # z = np.linspace(0, volume.shape[2], 1)
    # interpolating_function = RegularGridInterpolator((x, y, z), volume)

def find_best_z_iterative(image,ref, stride=20):
    moving_pcd = torch.from_numpy(volume_to_pointcloud(image, 8192, (300, 700)))
    emd_m = emdModule().cpu()
    best_z = 0
    best_score = float("infinity")
    for i in range(0, ref.shape[-1], stride):
        ref_pcd = torch.from_numpy(volume_to_pointcloud(ref[:, :, i:i+stride], 8192, (300, 700)))
        dis, ass = emd_m(moving_pcd, ref_pcd, 0.05, 3000)
        score = np.sqrt(dis.cpu()).mean()

        if score < best_score:
            best_score = score
            best_z = i

    return best_z

def find_best_z_cpu(image, ref, stride=10, window=8):
    moving_pcd = volume_to_pointcloud(image, int(8192/4), intensity_range=(300, 700))
    best_z = 0
    best_score = float("infinity")
    print("current best score is: " + str(best_score))
    for i in tqdm.tqdm(range(0, ref.shape[-1], stride)):
        ref_pcd = volume_to_pointcloud(ref[:, :, i:i+window], int(8192/4),intensity_range= (300, 700))
        if not ref_pcd:
            continue

        score = emd_scipy(np.asarray(moving_pcd.points), np.asarray(ref_pcd.points))
        if score < best_score:
            best_score = score
            best_z = i
        print("current best score is: " + str(best_score) + " with z of " + str(best_z))

    return best_z

if "__main__" == __name__:
    from data_process import *
    reference = load_file(r"D:\head-neck-clean\ct\HN-CHUM-014.nii.gz")
    moving = load_file(r"D:\head-neck-clean\ct\HN-CHUM-012.nii.gz")
    # angle = 30  # angle should be in degrees
    # moving = interpolation.rotate(reference[:,:,120:180], angle, reshape=False, prefilter=False)
    reference_pcd = volume_to_pointcloud(reference[:,:,60:60+8], int(8192/2), [1,0,0], (300, 700),)
    moving_pcd = volume_to_pointcloud(moving[:,:,60:60+8], int(8192/2), [0,1,0], (300, 700))


    # best_z = find_best_z_cpu(moving,reference)
    best_z = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    moving_pcd.points = o3d.utility.Vector3dVector(np.asarray(moving_pcd.points)+np.array([[0,0,best_z]]))
    vis.add_geometry(reference_pcd)
    vis.add_geometry(moving_pcd)
    vis.run()
    # while(True):
        # transform geometry using ICP
        # vis.poll_events()
    # pcl_a = volume_to_pointcloud(reference[:,:,460:520], 8192, (300, 700)).reshape(-1,3)
    # pcl_b = volume_to_pointcloud(moving, 8192, (300, 700)).reshape(-1,3)

    # o3d.visualization.draw_geometries([pcl_a, pcl_b ])

    # print(emd_scipy(pcl_a,pcl_b))
    # atlas = atlas.cpu().detach().numpy()[0,0,:,:,:]


    # o3d.visualization.draw_geometries([atlas_pcd])
