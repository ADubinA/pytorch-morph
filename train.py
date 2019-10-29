from network import *
from data_process import *
import metrics
import torch.optim as optim
from metrics import *
from tqdm import tqdm
import tools.tools as tools
from torch.autograd import Variable
import visualize
def train(atlas_path, train_dir, save_dir, sample_dir=None,
          split_tet=(0.8, 0.1, 0.1),
          images_per_epoch=50, epochs=100, learning_rate=0.0001, batch_size=1,
          save_interval=5, sample_interval=1):

    # set saving location for model and samples
    tools.set_path(save_dir)
    if sample_dir is not None:
        tools.set_path(sample_dir)

    atlas = load_file_for_stn(atlas_path)
    net = BilinearSTNRegistrator(atlas)
    net.cuda()
    dataset_gen = network_input(train_dir, split_tet, batch_size)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = metrics.loss_mse_with_grad

    running_loss = 0.0
    for epoch in tqdm(range(epochs)):
        for i in range(images_per_epoch):
            batch_data = next(dataset_gen)
            batch_data = Variable(batch_data, requires_grad=True)
            outputs = net(batch_data)
            loss = criterion(outputs, net.atlas)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))

        # sample after after save_interval epochs
        if epoch % sample_interval == 0:

            visualize.create_result(atlas=atlas, original=batch_data[0],
                                    vector_field=outputs[1][0],
                                    warped=outputs[0][0],
                                    save_location=tools.save_sample_string(sample_dir, epoch))

        # save model after save_interval epochs
        if epoch % save_interval == 0:
            torch.save(net.state_dict(), tools.save_model_string(save_dir, epoch))
        running_loss = 0.0

if __name__ == "__main__":
    linux_path = "/media/almog-lab/dataset/"
    win_path = r"D:/"
    sys_path = linux_path
    train(
        atlas_path=os.path.join(sys_path, "head-neck-reg-small/ct/HN-CHUM-007.nii.gz"),
        train_dir=os.path.join(sys_path, "head-neck-reg-small/ct"),
        save_dir=os.path.join(sys_path, "models/head-neck-reg-small"),
        sample_dir=os.path.join(sys_path, "models/head-neck-reg-small/samples")
    )