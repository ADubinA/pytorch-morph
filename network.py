import torch
import torch.nn as nn
import torch.nn.functional as F




class VoxelMorph1(nn.Module):
    """
    This is the Unet used by An Unsupervised Learning Model for
    Deformable Medical Image Registration
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Balakrishnan_An_Unsupervised_Learning_CVPR_2018_paper.pdf
    aka Voxelmorph1
    """
    def __init__(self):
        super().__init__()

        self.conv_layer_down1 = self.single_conv(1, 16)
        self.conv_layer_down2 = self.single_conv(16, 32)
        self.conv_layer_down32 = self.single_conv(32, 32)


        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv_layer_up32 = self.single_conv(32 + 32, 32)
        self.conv_layer_up1 = self.single_conv(32, 8)
        self.conv_layer_up2 = self.single_conv(8, 8)
        self.conv_layer_up3 = self.single_conv(8+16, 3)

        # self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        # 1: from 2 to 16 of scale 1
        conv1 = self.conv_layer_down1(x)
        x = self.maxpool(conv1)

        # 2: from 16 to 32 of scale 1/2
        conv2 = self.conv_layer_down2(x)
        x = self.maxpool(conv2)

        # 3: from 32 to 32 of scale 1/4
        conv3 = self.conv_layer_down32(x)
        x = self.maxpool(conv3)

        # # 4: from 32 to 32 of scale 1/8
        conv4 = self.conv_layer_down32(x)
        x = self.maxpool(conv4)

        # 5: from 32 to 32 of scale 1/16 ---- middle layer
        x = self.conv_layer_down32(x)

        # # 6: from 32+32 to 32 of scale 1/8
        x = self.upsample(x)
        x = torch.cat((x, conv4), dim=1)
        x = self.conv_layer_up32(x)

        # 7: from 32+32 to 32 of scale 1/4
        x = self.upsample(x)
        x = torch.cat((x, conv3), dim=1)
        x = self.conv_layer_up32(x)

        # 8: from 32+32 to 32 of scale 1/2
        x = self.upsample(x)
        x = torch.cat((x, conv2), dim=1)
        x = self.conv_layer_up32(x)

        # 9: from 32 to 8 of scale 1/2
        x = self.conv_layer_up1(x)

        # # 10: from 8 to 8 of scale 1
        x = self.upsample(x)
        # x = torch.cat((x, conv3), dim=1)
        x = self.conv_layer_up2(x)

        # 11: from 8+8 to 3 of scale 1
        x = torch.cat((x, conv1), dim=1)
        out = self.conv_layer_up3(x)

        return out
    @staticmethod
    def single_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
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

    def __init__(self,atlas):
        """

        Args:
            atlas:
                numpy array of size

        """
        super(BilinearSTNRegistrator, self).__init__()

        self.atlas = torch.from_numpy(atlas).float()
        self.localization_net = VoxelMorph1()

    def forward(self, x):
        """
        forward pass of the Bilinear STN registation using the atlas given in the constructor
        Args:
            x:

        Returns:

        """
        x = torch.from_numpy(x).float()
        vector_map = self.localization_net(torch.cat((self.atlas, x)))

        warped_image = F.grid_sample(input=x, grid=vector_map)

        return torch.cat((warped_image, vector_map))

