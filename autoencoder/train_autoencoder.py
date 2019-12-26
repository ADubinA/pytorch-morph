from autoencoder.network import AutoEncoderV1
from autoencoder.voxelmorph_losses import *
from data_process import *
import metrics
import torch.optim as optim
from metrics import *
from tqdm import tqdm
import tools.tools as tools
import visualize
import testing.debug_tools as debug_tools
from tensorboardX import SummaryWriter
import random

def train(train_dir, save_dir, sample_dir=None,tensorboard_dir=None, load_checkout_path=None,
          split_tet=(0.8, 0.1, 0.1),
          images_per_epoch=10, epochs=100, learning_rate=0.001, batch_size=1,
          save_interval=10, sample_interval=1):

    writer = SummaryWriter(os.path.join(tensorboard_dir, tools.save_string("",None)))

    # set saving location for model and samples
    tools.set_path(save_dir)
    if sample_dir is not None:
        tools.set_path(sample_dir)
    if tensorboard_dir is not None:
        tools.set_path(tensorboard_dir)

    # atlas = torch.tensor(skimage.transform.rescale(atlas,(1,1,0.5,0.5,0.5)))
    net = AutoEncoderV1()
    net.cuda()
    net.train()
    if load_checkout_path is not None:
        net.load_state_dict(torch.load(load_checkout_path))

    dataset_gen = network_input(train_dir, split_tet, batch_size)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = ncc_loss
    # writer.add_graph(net,next(dataset_gen))
    # writer.close()
    running_loss = 0.0
    for epoch in tqdm(range(epochs)):

        for i in range(images_per_epoch):
            optimizer.zero_grad()
            batch_data = next(dataset_gen)*0.0001
            # batch_data = torch.tensor(skimage.transform.rescale(batch_data.detach().cpu().numpy(), (1, 1, 0.5, 0.5, 0.5)))
            # -------test ---------------------

            # rand_shift = random.randint(-10, 10)
            # batch_data = debug_tools.rotate_vol(atlas, rand_shift)
            # batch_data = debug_tools.roll(batch_data,2,rand_shift).cuda()
            # ---------------------------------
            # batch_data = batch_data.cuda()
            # batch_data = Variable(batch_data, requires_grad=True)
            outputs = net(batch_data)
            loss = criterion(outputs, batch_data)*1000
            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

        # sample after after save_interval epochs
        if epoch % sample_interval == 0:

            for n, p in net.named_parameters():
                if p.requires_grad and "bias" not in n:
                    writer.add_histogram("gradients in hidden layer {}".format(n), p.grad, global_step=epoch)

            writer.add_scalar("total loss",loss, global_step=epoch)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.set_title("after autoencoder")
            ax1.imshow(outputs[0,0,:,:,15].cpu().detach().numpy())
            ax2.set_title("original")
            ax2.imshow(batch_data[0,0,:,:,15].cpu().detach().numpy())

            fig.savefig(tools.save_sample_string(sample_dir,epoch))

        # save model after save_interval epochs
        if epoch % save_interval == 0:
            torch.save(net.state_dict(), tools.save_model_string(save_dir, epoch))
        running_loss = 0.0

if __name__ == "__main__":
    linux_path = "/media/almog-lab/dataset/"
    win_path = r"D:/"
    sys_path = linux_path
    train(
        train_dir=os.path.join(sys_path, "head-neck-reg-small/ct"),
        save_dir=os.path.join(sys_path, "models/head-neck-reg-small"),
        sample_dir=os.path.join(sys_path, "models/head-neck-reg-small/samples"),
        tensorboard_dir=os.path.join(sys_path, "models/head-neck-reg-small/tensorboard_dir"),
        # load_checkout_path=os.path.join(sys_path, "models/head-neck-reg-small", "20191111-130805_epoch-40.pt")
    )