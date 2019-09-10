from network import *
from data_process import *
import metrics
import torch.optim as optim
from metrics import loss_fn
import matplotlib.pyplot as plt


def train(atlas_path, train_dir, split_tet=(0.8,0.1,0.1), images_per_epoch=10, epochs=10, learning_rate=0.001, batch_size=1):

    atlas = load_file_for_stn(atlas_path)
    net = BilinearSTNRegistrator(atlas)
    dataset_gen = network_input(train_dir, split_tet, batch_size)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = metrics.MSE_loss

    running_loss = 0.0
    for epoch in range(epochs):
        for i in range(images_per_epoch):
            batch_data = next(dataset_gen)
            outputs = net(batch_data)[0]
            loss = criterion(outputs, atlas)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

if __name__ == "__main__":

    train(
        atlas_path=r"D:\head-neck-reg-small\ct\HN-CHUM-007.nii.gz",
        train_dir= r"D:\head-neck-reg-small\ct"
    )