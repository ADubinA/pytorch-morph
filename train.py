from network import *
from data_process import *
import torch.optim as optim
from metrics import *
from tqdm import tqdm
import tools.tools as tools
import time
from torch.autograd import Variable
import visualize
from tensorboardX import SummaryWriter
import torch.functional as F


def test_mask_model(model, atlas_name, test_path,  load_checkout_path=None):
    labels_for_testing = ["mask_BODY", "mask_Brainstem", "mask_Brain"]
    result_dict = {label : [] for label in labels_for_testing}
    if load_checkout_path is not None:
        model.load_state_dict(torch.load(load_checkout_path))
    gen = ct_pet_data_generator(test_path, "test", load_labels=True, labels=labels_for_testing, load_specific_files=['HN-HGJ-074'])
    atlas_data, atlas_labels = ct_pet_data_loader(test_path, "ct", atlas_name,
                                                  load_labels=True, labels=labels_for_testing)
    i = 1
    for test_example_data, test_example_labels in gen:
        test_example_warped, test_mask = model(test_example_data)

        for label_name, label_data in test_example_labels[0].items():
            slicing_square = to_slice(test_mask, atlas_data.shape[2:])
            dice_score = dice_loss(atlas_labels[label_name][slicing_square], label_data[slicing_square])
            result_dict[label_name].append(dice_score)

        i += 1
        if i == len(labels_for_testing):
            break

    result_dict = {label_name: np.mean(np.array(label_data)) for label_name, label_data in result_dict.items()}

    return result_dict


def test_stn_model(model, atlas_name, test_path,  load_checkout_path=None):
    labels_for_testing = ["BODY", "Brainstem", "Brain"]
    result_dict = {label : [] for label in labels_for_testing}
    if load_checkout_path is not None:
        model.load_state_dict(torch.load(load_checkout_path))
    gen = ct_pet_data_generator(test_path, "test", load_labels=True, labels=labels_for_testing)
    atlas_data, atlas_labels = ct_pet_data_loader(test_path, "ct", atlas_name,
                                                  load_labels=True, labels=labels_for_testing)

    for test_example_data, test_example_labels in gen:
        test_example_warped, test_example_grid = model(test_example_data)

        for label_name, label_data in test_example_labels[0].items():
            warped_label = F.grid_sample(input=label_data,
                                         grid=test_example_grid, padding_mode="zero")
            dice_score = dice_loss(atlas_labels[label_name], warped_label)
            result_dict[label_name].append(dice_score)

    result_dict = {label_name: np.mean(np.array(label_data)) for label_name, label_data in result_dict.items()}

    return result_dict


def train(atlas_name, train_dir, save_dir, sample_dir=None,tensorboard_dir=None, load_checkout_path=None,
          images_per_epoch=15, epochs=40, learning_rate=0.001, batch_size=1,
          save_interval=1, sample_interval=1):

    writer = SummaryWriter(os.path.join(tensorboard_dir, tools.save_string("",None)))

    # set saving location for model and samples
    tools.set_path(save_dir)
    if sample_dir is not None:
        tools.set_path(sample_dir)
    if tensorboard_dir is not None:
        tools.set_path(tensorboard_dir)

    atlas = ct_pet_data_loader(train_dir, "ct", atlas_name)[0]
    net = Type2Module(atlas, "cuda:0")
    net.cuda()
    net.train()
    if load_checkout_path is not None:
        net.load_state_dict(torch.load(load_checkout_path))

    dataset_gen = ct_pet_data_generator(train_dir, "train", load_specific_files=['HN-HGJ-074'])
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = pixel_loss_with_masking
    # writer.add_graph(net,next(dataset_gen))
    # writer.close()
    running_loss = 0.0
    for epoch in tqdm(range(epochs)):

        for i in range(images_per_epoch):
            optimizer.zero_grad()
            batch_data = next(dataset_gen)
            # batch_data = torch.tensor(skimage.transform.rescale(batch_data.detach().cpu().numpy(), (1, 1, 0.5, 0.5, 0.5)))
            # -------test ---------------------

            # rand_shift = random.randint(-10, 10)
            # batch_data = debug_tools.rotate_vol(atlas, rand_shift)
            # batch_data = debug_tools.roll(atlas,2,rand_shift).cuda()

            # batch_data = batch_data.cuda()
            # batch_data = Variable(batch_data, requires_grad=True)

            # batch_data = tools.random_image_slice(atlas, (0,0,0), (80,80,20))
            # ---------------------------------
            outputs = net(batch_data)
            loss = criterion(outputs[0], outputs[1], net.atlas) + 0.0005*((entropy_loss(atlas)-entropy_loss(outputs[0])).mean())**2 #+ 100*mask_affine_regularization(outputs[1], net.atlas.shape[2:])
            loss.backward()
            optimizer.step()


            # debug_tools.plot_grad_flow(net.named_parameters())
            # print statistics
            running_loss += loss.data[0]

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
            print(outputs[1])

        # sample after after save_interval epochs
        if epoch % sample_interval == 0:

            # for n, p in net.named_parameters():
            #     if p.requires_grad and "bias" not in n:
            #         writer.add_histogram("gradients in hidden layer {}".format(n), p.grad, global_step=epoch)
            #
            # writer.add_scalar("total loss",loss, global_step=epoch)

            visualize.create_3d_result(atlas=net.affine_stn.atlas, original=batch_data[0],
                                    vector_field=outputs[1][0],
                                    warped=outputs[0][0],
                                    save_location=tools.save_sample_string(sample_dir, epoch))

        # save model after save_interval epochs
        if epoch % save_interval == 0:
            torch.save(net.state_dict(), tools.save_model_string(save_dir, epoch))
        running_loss = 0.0

    print(test_mask_model(net, atlas_name, train_dir))

if __name__ == "__main__":
    linux_path = "/media/almog-lab/dataset/"
    win_path = r"D:/"
    sys_path = win_path
    train(
        atlas_name="HN-HGJ-077",
        train_dir=os.path.join(sys_path, "head-neck-clean"),
        save_dir=os.path.join(sys_path, "models/saves", time.strftime("%Y%m%d-%H%M%S")),
        sample_dir=os.path.join(sys_path, "models/samples", time.strftime("%Y%m%d-%H%M%S")),
        tensorboard_dir=os.path.join(sys_path, "models/head-neck-reg-small/tensorboard_dir"),
        # load_checkout_path=os.path.join(sys_path, "models/head-neck-reg-small", "20191111-130805_epoch-40.pt")
    )