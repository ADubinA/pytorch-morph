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

def create_result(atlas,original, warped, vector_field, save_location):

    slice_index = 15
    dim = 2

    original = original[0, :, :, :].cpu().detach().numpy()
    warped = warped[0, :, :, :].cpu().detach().numpy()
    atlas = atlas[0,0, :, :, :].cpu().detach().numpy()

    slicer = [slice(None)]*3
    slicer[dim] = slice_index

    original = original[slicer]
    warped = warped[slicer]
    atlas = atlas[slicer]

    fig, axs = plt.subplots(1, 4, figsize=(9, 3), sharex=True)
    axs[0].imshow(original)
    axs[0].title.set_text("original image")

    axs[1].imshow(atlas)
    axs[1].title.set_text("atlas")
    axs[2].imshow(warped)
    axs[2].title.set_text("warped image")


    plt_2d_vector_field_tensor(vector_field,slice_index,dim,None,axs[3])
    axs[3].title.set_text("vector field")

    fig.savefig(save_location)

def plt_2d_vector_field_tensor(vector_field, slicing_index, dim=0, save_location=None,ax=None):
    vector_field = vector_field.cpu()
    vector_field = vector_field.detach().numpy()
    return plt_2d_vector_field(vector_field,slicing_index,dim,save_location,ax=ax)

def plt_2d_vector_field(vector_field, slicing_index, dim=0, save_location=None, skip_resolution=1,ax=None):
    """
    drawing a 2d vector field generated from the net
    Args:
        vector_field: (numpy ndarray)
            the vector field, must have shape of the form (x,y,z,3)
        slicing_index: (int)
            slice location for the vector field
        dim:(int)
            dimension index for slicing
        save_location: (string)
            full location for saving the image. if None, will use plt.show()


    Returns:
        None
    """
    vector_field = vector_field

    # remove the slicing dim
    dims = [0,1,2]
    dims.pop(dim)
    # create a grid for plt
    x_grid = np.arange(0, vector_field.shape[dims[1]])
    y_grid = np.arange(0, vector_field.shape[dims[0]])
    # y_grid, x_grid = np.meshgrid(x_grid, y_grid) # yes, x and y are in different order here
    if ax is None:
        fig0, ax = plt.subplots()

    # set slicing parameters
    slicer = [slice(None)]*4
    slicer[dim] = slicing_index

    # slice the vector field on the right dim
    flat_vector_field = vector_field[slicer]
    # reshape from (shape,3) to (3, shape)
    flat_vector_field = flat_vector_field.reshape([3]+list(flat_vector_field.shape)[:-1])
    x = flat_vector_field[dims[0]]
    y = flat_vector_field[dims[1]]

    # q = ax0.quiver(x_grid, y_grid, x, y)


    ax.set_title("pivot='tip'; scales with x view")
    # M = np.hypot(x[:: skip_resolution, :: skip_resolution], y[:: skip_resolution, :: skip_resolution])
    q = ax.quiver(x_grid[:: skip_resolution],
                   y_grid[::skip_resolution],
                   x[:: skip_resolution, :: skip_resolution],
                   y[:: skip_resolution, :: skip_resolution])
                   # M,
                   # units='x', pivot='tip', width=0.022, scale=1 / 0.15)
    ax.quiverkey(q, X=0.3, Y=1.1, U=10,
                 label='Quiver key, length = 10', labelpos='E')
    # ax0.scatter(x_grid, y_grid, color='0.5', s=1)
    # # , color=x
    # strm = ax0.streamplot(x_grid, y_grid, x, y, linewidth=2, cmap=plt.cm.autumn)
    # fig0.colorbar(strm.lines)
    #
    # fig1, (ax1, ax2) = plt.subplots(ncols=2)
    # ax1.streamplot(x_grid, y_grid, x, y, density=[0.5, 1])
    #
    # lw = 1 #5 * speed / speed.max()
    # ax2.streamplot(x_grid, y_grid, x, y, density=0.6, color='k', linewidth=lw)
    if save_location is None:
        return
    else:
        plt.savefig(save_location)

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