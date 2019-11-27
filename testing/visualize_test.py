from visualize import *
import torch.nn.functional as F
import data_process



def test_plot_grid():
    unit_grid = tools.create_unit_grid((128,128,64))
    test_grid = unit_grid + 1
    fig, ax = plt.subplots()
    plot_grid(test_grid[0].numpy(),15,dim=2, ax=ax)
    plt.show()

def test_plt_2d_vector_field():
    unit_grid = tools.create_unit_grid((128,128,64))
    test_grid = unit_grid + 1
    fig, ax = plt.subplots()
    plt_2d_vector_field(test_grid[0].numpy(),15, ax=ax)
    plt.show()

def find_unit_grid(file_location):
    mini = 10000000000000
    itera = -1
    image = data_process.load_file_for_stn(file_location)

    for i in range(1550,1600):
        i = i/100000
        print(i)
        unit_grid = tools.create_unit_grid((128,128,64))+i
        # test_grid = unit_grid + 0.25
        warp = F.grid_sample(image,unit_grid)
        if mini > (warp - image).sum():
            mini = (warp - image).sum()
            print("mini = {}".format(i))
            itera = i
    # plt.imshow((warp-image).cpu().detach().numpy()[0,0,:,:,15])
    # plt.show()

def test_unit_grid(file_location):
    image = data_process.load_file_for_stn(file_location)

    unit_grid = tools.create_unit_grid((128,128,64))+0.01563
    # test_grid = unit_grid + 0.25
    warp = F.grid_sample(image,unit_grid)

    plt.imshow((warp-image).cpu().detach().numpy()[0,0,:,:,15])
    plt.show()

if __name__ == '__main__':
    # test_unit_grid("/home/almog-lab/dev/pytorch-morph/data/test.nii.gz")
    test_plot_grid()