from visualize import *

def test_plt_2d_vector_field():
    # Set limits and number of points in grid
    grid = np.mgrid[10:-10:100j, 10:-10:100j,10:-10:100j]
    p = np.sqrt(grid[0]+grid[1] ** 2 + grid[2] ** 3)
    grad = np.array(np.gradient(p))
    grad_size = list(grad.shape)[1:]
    grad_size.append(3)
    grad = grad.reshape(grad_size)

    plt_2d_vector_field(grad,1)

if __name__ == '__main__':
    test_plt_2d_vector_field()
