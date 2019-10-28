import numpy as np
# from vispy import app, scene
# from multivol import MultiVolume
# from multivol import get_translucent_cmap
# from vispy import io, plot as vp
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import math
# import mayavi.mlab as mlab

def threshold(data, data_min, data_max):
    data[data > data_max] = 0
    data[data < data_min] = 0
    return data

def normalize(vol_data):
    vol_data-= min(0,np.min(vol_data))
    vol_data = vol_data.astype("float64")
    vol_data *= 255.0/vol_data.max()
    return vol_data.astype("int16")


def show_3d(volume, vol_min=-float("inf"),vol_max=float("inf")):

    fig = vp.Fig(bgcolor='k', size=(800, 800), show=False)
    vol_data = volume
    vol_data = np.flipud(np.rollaxis(vol_data, 1))

    vol_data = threshold(vol_data, vol_min, vol_max)

    clim = [32, 192]
    vol_pw = fig[0, 0]

    vol_data -= min(0, np.min(vol_data))
    vol_data = vol_data.astype("float64")
    vol_data *= 255.0/vol_data.max()

    vol_pw.volume(vol_data, clim=clim,  cmap='hot')
    vol_pw.camera.elevation = 30
    vol_pw.camera.azimuth = 30
    vol_pw.camera.scale_factor /= 1.5

    shape = vol_data.shape
    fig[1, 0].image(vol_data[:, :, shape[2] // 2],   cmap='hot', clim=clim)
    fig[0, 1].image(vol_data[:, shape[1] // 2, :],   cmap='hot', clim=clim)
    fig[1, 1].image(vol_data[shape[0] // 2, :, :].T, cmap='hot', clim=clim)
    fig.show(run=True)

def show_merge_3d(volume0, volume1, vol_min=-float("inf"),vol_max=float("inf")):

    fig = vp.Fig(bgcolor='k', size=(800, 800), show=False)

    volume0 = threshold(volume0, vol_min, vol_max)
    volume1 = threshold(volume1, vol_min, vol_max)

    volume0 = normalize(volume0)
    volume1 = normalize(volume1)

    # Prepare canvas
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
    canvas.measure_fps()

    # Set up a viewbox to display the image with interactive pan/zoom
    view = canvas.central_widget.add_view()

    # Set whether we are emulating a 3D texture
    emulate_texture = False

    reds = get_translucent_cmap(1, 0, 0)
    blues = get_translucent_cmap(0, 0, 1)

    # Create the volume visuals, only one is visible
    volumes = [(volume0, None, blues), (volume1, None, reds)]
    volume = MultiVolume(volumes, parent=view.scene, threshold=0.225, emulate_texture=emulate_texture)
    volume.transform = scene.STTransform(translate=(64, 64, 0))

    # Create three cameras (Fly, Turntable and Arcball)
    fov = 60.
    cam2 = scene.cameras.TurntableCamera(parent=view.scene, fov=fov, name='Turntable')
    view.camera = cam2  # Select turntable at first

    canvas.update()
    app.run()

def show_difference_2d(volume0, volume1,slice_dim=0 ,jump=1, vol_min=-float("inf"),vol_max=float("inf")):
    indx = [slice(None)] * volume0.ndim
    indx[slice_dim] = slice(0, min(volume0.shape[slice_dim],volume0.shape[slice_dim]),jump)

    volume0_slice = volume0[tuple(indx)]
    volume1_slice = volume1[tuple(indx)]

    num_of_images = volume0_slice.shape[slice_dim]
    images_per_row = 5
    fig, ax = plt.subplots(math.ceil(num_of_images/images_per_row)-1, images_per_row, sharex='col', sharey='row')



    for i in range(math.ceil(num_of_images/images_per_row)-1):
        for j in range(images_per_row):
            indx[slice_dim] = i*images_per_row+j
            ax[i, j].imshow(volume0_slice[tuple(indx)])

    plt.figure()
    fig, ax = plt.subplots(math.ceil(num_of_images/images_per_row)-1, images_per_row, sharex='col', sharey='row')



    for i in range(math.ceil(num_of_images/images_per_row)-1):
        for j in range(images_per_row):
            indx[slice_dim] = i*images_per_row+j
            ax[i, j].imshow(volume1_slice[tuple(indx)])
    plt.show()

def show_two_2d(volume0, volume1,index,slice_dim=0, vol_min=-float("inf"),vol_max=float("inf")):

    indx = [slice(None)] * volume0.ndim
    indx[slice_dim] = index


    plt.imshow(volume0[tuple(indx)])

    plt.figure()
    indx[slice_dim] = index
    plt.imshow(volume1[tuple(indx)])
    plt.show()

def show_merge_2d(volume0, volume1,slice_dim=0 ,jump=1, vol_min=-float("inf"),vol_max=float("inf")):
    indx = [slice(None)] * volume0.ndim
    indx[slice_dim] = slice(0, min(volume0.shape[slice_dim],volume0.shape[slice_dim]),jump)

    volume0_slice = volume0[tuple(indx)]
    volume1_slice = volume1[tuple(indx)]

    num_of_images = volume0_slice.shape[slice_dim]
    images_per_row = 5
    fig, ax = plt.subplots(math.ceil(num_of_images/images_per_row), images_per_row+1, sharex='col', sharey='row')

    indx[slice_dim] = slice(0,3)
    img = np.zeros(shape=(128,128,3))
    for i in range(math.floor(num_of_images/images_per_row)):
        for j in range(images_per_row):
            indx[slice_dim] = i*images_per_row+j
            img[:,:,0] = normalize(volume0_slice[tuple(indx)])
            img[:,:,1] = normalize(volume1_slice[tuple(indx)])
            # ax[i,j].imshow(img)
            # ax[i, j].imshow(volume0_slice[tuple(indx)], alpha=.8, interpolation='bilinear', cmap="Reds")
            # ax[i, j].imshow(volume1_slice[tuple(indx)], alpha=.8, interpolation='bilinear', cmap="Blues")

            # img[:,:,0] = normalize(volume0_slice[tuple(indx)]) + normalize(volume1_slice[tuple(indx)])
            ax[i,j].imshow(img)
    plt.show()


def show_vector_field(volume):
    u = volume[..., 0]
    v = volume[..., 1]
    w = volume[..., 2]
    # mlab.quiver3d(u, v, w)
    # mlab.outline()
    #
    src = mlab.pipeline.vector_field(u, v, w)
    mlab.pipeline.vectors(src, mask_points=20, scale_factor=3.)

def plt_2d_vector_field(vector_field, slice, dim=0, save_location=None):
    """
    drawing a 2d vector field generated from the net
    Args:
        vector_field: (numpy ndarray)
            the vector field, must have shape of the form (x,y,z,3)
        slice: (int)
            slice location for the vector field
        dim:(int)
            dimension index for slicing
        save_location: (string)
            full location for saving the image. if None, will use plt.show()


    Returns:
        None
    """
    # Set limits and number of points in grid
    y, x = np.mgrid[10:-10:100j, 10:-10:100j]

    x_obstacle, y_obstacle = 5, 5
    alpha_obstacle, a_obstacle, b_obstacle = -1.0, 1e3, 2e3

    p = -alpha_obstacle * np.exp(-((x - x_obstacle) ** 2 / a_obstacle
                                   + (y - y_obstacle) ** 2 / b_obstacle))
    dy, dx = np.gradient(p)

    fig, ax = plt.subplots()

    ax.streamplot(x, y, dx, dy, linewidth=500 * np.hypot(dx, dy),
                  color=p, density=1.2, cmap='gist_earth')

    cont = ax.contour(x, y, p, cmap='gist_earth', vmin=p.min(), vmax=p.max())
    labels = ax.clabel(cont)

    plt.setp(labels, path_effects=[withStroke(linewidth=8, foreground='w')])

    ax.set(aspect=1, title='Streamplot with contours')
    plt.show()

def show_histogram(volume):
    plt.hist(volume.flatten(), bins='auto')  # arguments are passed to np.histogram

    plt.title("Histogram with 'auto' bins")
    plt.show()

if __name__ == '__main__':
    vol0 = np.load(r"D:/LIDC-IDRI_npz_small/0.npz")['arr_0']
    vol1 = np.load(r"D:/LIDC-IDRI_npz_small/1.npz")['arr_0']
    # vol1 = np.load(r"D:/output.npz")['arr_0']
    # vol1 = np.load(io.load_data_file(r"D:/small_register/0_moved.npz"))['arr_0']
    #show_merge_3d(vol0[:16,:,:],vol1, 1500)
    # show_difference_2d(vol0[:16,:,:], vol1,slice_dim=0 ,jump=1)
    show_two_2d(vol0[:16,:,:], vol1,5)
    # show_histogram(vol0)