import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import testing.debug_tools as debug_tools
import tools.tools as tools
# class VoxelMorph1(nn.Module):
#     """
#     This is the Unet used by An Unsupervised Learning Model for
#     Deformable Medical Image Registration
#     http://openaccess.thecvf.com/content_cvpr_2018/papers/Balakrishnan_An_Unsupervised_Learning_CVPR_2018_paper.pdf
#     aka Voxelmorph1
#     """
#     def __init__(self):
#         super().__init__()
#
#         self.conv_layer_down1 = self.single_conv(1, 16)
#         self.conv_layer_down2 = self.single_conv(16, 32)
#         self.conv_layer_down32 = self.single_conv(32, 32)
#
#
#         self.maxpool = nn.MaxPool3d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
#
#         self.conv_layer_up32 = self.single_conv(32 + 32, 32)
#         self.conv_layer_up1 = self.single_conv(32, 8)
#         self.conv_layer_up2 = self.single_conv(8, 8)
#         self.conv_layer_up3 = self.single_conv(8+16, 3)
#
#         # self.conv_last = nn.Conv2d(64, n_class, 1)
#
#     def forward(self, x):
#         # 1: from 2 to 16 of scale 1
#         conv1 = self.conv_layer_down1(x)
#         x = self.maxpool(conv1)
#
#         # 2: from 16 to 32 of scale 1/2
#         conv2 = self.conv_layer_down2(x)
#         x = self.maxpool(conv2)
#
#         # 3: from 32 to 32 of scale 1/4
#         conv3 = self.conv_layer_down32(x)
#         x = self.maxpool(conv3)
#
#         # # 4: from 32 to 32 of scale 1/8
#         conv4 = self.conv_layer_down32(x)
#         x = self.maxpool(conv4)
#
#         # 5: from 32 to 32 of scale 1/16 ---- middle layer
#         x = self.conv_layer_down32(x)
#
#         # # 6: from 32+32 to 32 of scale 1/8
#         x = self.upsample(x)
#         x = torch.cat((x, conv4), dim=1)
#         x = self.conv_layer_up32(x)
#
#         # 7: from 32+32 to 32 of scale 1/4
#         x = self.upsample(x)
#         x = torch.cat((x, conv3), dim=1)
#         x = self.conv_layer_up32(x)
#
#         # 8: from 32+32 to 32 of scale 1/2
#         x = self.upsample(x)
#         x = torch.cat((x, conv2), dim=1)
#         x = self.conv_layer_up32(x)
#
#         # 9: from 32 to 8 of scale 1/2
#         x = self.conv_layer_up1(x)
#
#         # # 10: from 8 to 8 of scale 1
#         x = self.upsample(x)
#         # x = torch.cat((x, conv3), dim=1)
#         x = self.conv_layer_up2(x)
#
#         # 11: from 8+8 to 3 of scale 1
#         x = torch.cat((x, conv1), dim=1)
#         out = self.conv_layer_up3(x)
#
#         return out
#     @staticmethod
#     def single_conv(in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             # nn.Conv3d(out_channels, out_channels, self.num_channels, padding=1),
#             # nn.ReLU(inplace=True)
#         )


class Type1Module(nn.Module):
    """
    This is the Unet used by An Unsupervised Learning Model for
    Deformable Medical Image Registration
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Balakrishnan_An_Unsupervised_Learning_CVPR_2018_paper.pdf
    aka Voxelmorph1
    """

    def __init__(self, atlas, device=None):
        super().__init__()

        self.conv_layer_down1 = self.single_conv(1, 16)
        self.conv_layer_down2 = self.single_conv(16, 32)
        self.conv_layer_down31 = self.single_conv(32, 32)
        self.conv_layer_down32 = self.single_conv(32, 32)
        self.conv_layer_down33 = self.single_conv(32, 32)

        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv_layer_up33 = self.single_conv(32 + 32, 32)
        self.conv_layer_up32 = self.single_conv(32 + 32, 32)
        self.conv_layer_up31 = self.single_conv(32 + 32, 32)

        self.conv_layer_up1 = self.single_conv(32, 8)
        self.conv_layer_up2 = self.single_conv(8, 8)

        # no relu at the end
        self.conv_layer_up3 = nn.Conv3d(in_channels=8 + 16, out_channels=3, kernel_size=3, padding=1)

        # self.conv_last = nn.Conv2d(64, n_class, 1)
        def init_normal(m):
            if type(m) == nn.Conv3d:
                nn.init.uniform_(m.weight)

        # use the modules apply function to recursively apply the initialization
        self.apply(init_normal)

        if atlas.type is np.ndarray:
            self.atlas = torch.from_numpy(atlas)
        else:
            self.atlas = atlas

        # set default device if needed
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.atlas = self.atlas.to(device)
        # apply the device
        self.unit_gird = tools.create_unit_grid(atlas.shape[2:]).to(device)

    def forward(self, x):
        original_image = x
        x = torch.cat((self.atlas, x))
        # 1: from 2 to 16 of scale 1
        conv1 = self.conv_layer_down1(x)
        x = self.maxpool(conv1)
        conv1 = conv1[1:,...]  # remove atlas so skip connection will work proper


        # 2: from 16 to 32 of scale 1/2
        conv2 = self.conv_layer_down2(x)
        x = self.maxpool(conv2)
        conv2 = conv2[1:,...]  # remove atlas so skip connection will work proper


        # 3: from 32 to 32 of scale 1/4
        conv3 = self.conv_layer_down31(x)
        x = self.maxpool(conv3)
        conv3 = conv3[1:,...]  # remove atlas so skip connection will work proper


        # # 4: from 32 to 32 of scale 1/8
        conv4 = self.conv_layer_down32(x)
        x = self.maxpool(conv4)
        conv4 = conv4[1:,...]  # remove atlas so skip connection will work proper

        # 5: from 32 to 32 of scale 1/16 ---- middle layer
        x = self.conv_layer_down33(x)

        # add course atlas to course batch
        x = x[1:,...] + x[0]

        # # 6: from 32+32 to 32 of scale 1/8
        x = self.upsample(x)
        x = torch.cat((x, conv4), dim=1)
        x = self.conv_layer_up33(x)

        # 7: from 32+32 to 32 of scale 1/4
        x = self.upsample(x)
        x = torch.cat((x, conv3), dim=1)
        x = self.conv_layer_up32(x)

        # 8: from 32+32 to 32 of scale 1/2
        x = self.upsample(x)
        x = torch.cat((x, conv2), dim=1)
        x = self.conv_layer_up31(x)

        # 9: from 32 to 8 of scale 1/2
        x = self.conv_layer_up1(x)

        # # 10: from 8 to 8 of scale 1
        x = self.upsample(x)
        # x = torch.cat((x, conv3), dim=1)
        x = self.conv_layer_up2(x)

        # 11: from 8+8 to 3 of scale 1
        x = torch.cat((x, conv1), dim=1)
        out = self.conv_layer_up3(x)
        out = 2 * torch.sigmoid(out) - 1
        out = out.permute(0,2,3,4,1)

        vector_map = out + self.unit_gird
        warped_image = F.grid_sample(input=original_image, grid=vector_map)
        return warped_image, vector_map


    @staticmethod
    def single_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            # nn.Conv3d(out_channels, out_channels, self.num_channels, padding=1),
            # nn.ReLU(inplace=True)
        )


class BilinearSTNRegistrator(nn.Module):
    """
    3D spatial transformer implementation using pytorch.
    the STN has 3 main parts:
        1. The localization network (who is giving by to user at init level)
        2. creating are transformation function. This could be affine, b-slin
    """

    def __init__(self, atlas, device=None):
        """

        Args:
            atlas:
                numpy array of size
            using_cuda(bool)

        """
        super(BilinearSTNRegistrator, self).__init__()

        if atlas.type is np.ndarray:
            self.atlas = torch.from_numpy(atlas)
        else:
            self.atlas = atlas
        self.atlas = self.atlas.to(device)
        self.atlas = Variable(self.atlas, requires_grad=True)

        self.localization_net = Type1Module()

        # set default device if needed
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        # apply the device
        self.localization_net.to(device)
        self.unit_gird = tools.create_unit_grid(atlas.shape[2:]).to(device)
    def forward(self, x):
        """
        forward pass of the Bilinear STN registation using the atlas given in the constructor
        Args:
            x:

        Returns:

        """
        localization_input = torch.cat((self.atlas, x))
        # localization_input = localization_input.to(self.device)
        vector_map = self.localization_net(localization_input) + self.unit_gird
        # vector_map = tools.create_unit_grid(x.shape[2:])
        # vector_map = vector_map.to(self.device)
        warped_image = F.grid_sample(input=x, grid=vector_map, padding_mode="reflection")

        return (warped_image, vector_map)

