from network import *
from data_process import *
import torch.optim as optim
from metrics import loss_fn
def train(atlas_path, train_dir, split_tuple=(0.8,0.1,0.1), epochs=10, learning_rate=0.001):

    atlas = load_file(atlas_path)
    net = BilinearSTNRegistrator(atlas)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        pass