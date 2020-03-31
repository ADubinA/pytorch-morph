from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
from torch.autograd import Variable
import numpy as np

import testing.debug_tools as debug_tools
import tools.tools as tools

class Masknet(nn.Module):
    def __init__(self):
        super().__init__()


# -Localization modules ------------------------------------------------------------------------------------------------
class LocalizationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    @staticmethod
    def single_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            # nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.ReLU(inplace=False)
        )

class UnetLocalizationNet(LocalizationNet):

    def __init__(self) -> None:
        super().__init__()
        self.conv_layer_down1 = self.single_conv(1, 16)
        self.conv_layer_down2 = self.single_conv(16, 32)
        self.conv_layer_down31 = self.single_conv(32, 32)
        self.conv_layer_down32 = self.single_conv(32, 32)
        self.conv_layer_down33 = self.single_conv(32, 32)

        self.conv_layer_up33 = self.single_conv(32 + 32, 32)
        self.conv_layer_up32 = self.single_conv(32 + 32, 32)
        self.conv_layer_up31 = self.single_conv(32 + 32, 32)

        self.conv_layer_up1 = self.single_conv(32, 8)
        self.conv_layer_up2 = self.single_conv(8, 8)

        # no relu at the end
        self.conv_layer_up3 = nn.Conv3d(in_channels=8 + 16, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):

        # 1: from 2 to 16 of scale 1
        conv1 = self.conv_layer_down1(x)
        x = self.maxpool(conv1)
        conv1 = conv1[1:, ...]  # remove atlas so skip connection will work proper

        # 2: from 16 to 32 of scale 1/2
        conv2 = self.conv_layer_down2(x)
        x = self.maxpool(conv2)
        conv2 = conv2[1:, ...]  # remove atlas so skip connection will work proper

        # 3: from 32 to 32 of scale 1/4
        conv3 = self.conv_layer_down31(x)
        x = self.maxpool(conv3)
        conv3 = conv3[1:, ...]  # remove atlas so skip connection will work proper

        # # # 4: from 32 to 32 of scale 1/8
        conv4 = self.conv_layer_down32(x)
        # x = self.maxpool(conv4)
        conv4 = conv4[1:, ...]  # remove atlas so skip connection will work proper

        # 5: from 32 to 32 of scale 1/16 ---- middle layer
        x = self.conv_layer_down33(x)

        # add course atlas to course batch
        x = x[1:, ...] + x[0]

        # # # 6: from 32+32 to 32 of scale 1/8
        # x = self.upsample(x)
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
        x = self.conv_layer_up2(x)

        # 11: from 8+8 to 3 of scale 1
        x = torch.cat((x, conv1), dim=1)
        x = self.conv_layer_up3(x)
        out = 2 * torch.sigmoid(x) - 1
        return out


class AffineLocalizationNet(LocalizationNet):

    def __init__(self, fc_size) -> None:
        super().__init__()

        self.fc_size = fc_size

        self.convnet = nn.Sequential(
            self.single_conv(1, 16), self.maxpool,
            self.single_conv(16, 32), self.maxpool,
            self.single_conv(32, 32), self.maxpool,
            self.single_conv(32, 32), self.maxpool,
            )

        self.fc_net = nn.Sequential(
            nn.Linear(self.fc_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 4 * 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_net[2].weight.data.zero_()
        self.fc_net[2].bias.data.copy_(torch.tensor([1, 0, 0, 0,
                                                     0, 1, 0, 0,
                                                     0, 0, 1, 0], dtype=torch.float))


    def forward(self, x):

        x = self.convnet(x)
        x = x.view(-1, self.fc_size)

        theta = self.fc_net(x)
        theta = theta.view(-1, 3, 4)

        return theta

# ----------------------------------------------------------------------------------------------------------------------

# -STN modules----------------------------------------------------------------------------------------------------------


class AbsSTN(nn.Module):
    """
    This is the Unet used by An Unsupervised Learning Model for
    Deformable Medical Image Registration
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Balakrishnan_An_Unsupervised_Learning_CVPR_2018_paper.pdf
    aka Voxelmorph1
    """

    def __init__(self, atlas, device=None):
        super().__init__()
        self.localization_net = nn.Module
        self.atlas = atlas.to(device)
        self.unit_gird = tools.create_unit_grid(atlas.shape[2:]).to(device)


class GridSTN(AbsSTN):
    def __init__(self, atlas, device=None):
        super().__init__(atlas, device)
        self.localization_net = UnetLocalizationNet()
        print(torch.cuda.memory_allocated())

    def forward(self, original_image):
        x = torch.cat((self.atlas, original_image))
        vector_map = self.localization_net(x)
        grid = vector_map.permute(0, 2, 3, 4, 1) + self.unit_gird
        warped_image = F.grid_sample(input=original_image, grid=grid, padding_mode="reflection")
        return warped_image, grid


class AffineSTN(AbsSTN):
    def __init__(self, atlas, device=None):
        super().__init__(atlas, device)
        # for 255*255: 32*16*16*2
        # for 128*128: 32*16*4*2
        self.localization_net = AffineLocalizationNet(32*16*4*2)

    def forward(self, x):
        theta = self.localization_net(x)
        grid = F.affine_grid(theta, x.size())
        warped_image = F.grid_sample(x, grid)
        return warped_image, grid


class Type1Module(nn.Module):
    def __init__(self, atlas, device=None):
        super().__init__()
        self.affine_stn = AffineSTN(atlas, device)
        self.grid_stn = GridSTN(atlas, device)
        self.atlas = atlas
    def forward(self, x):
        x, affine_theta = self.affine_stn(x)
        x, affine_theta = self.affine_stn(x)

        x, grid = self.grid_stn(x)
        # x, grid = self.grid_stn(x)
        # x, grid = self.grid_stn(x)
        # print(torch.cuda.memory_allocated())

        return x, grid
