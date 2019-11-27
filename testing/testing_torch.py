import torch
import torch.nn as nn
from tools.tools import batch_duplication
import torch.nn.functional as F
import matplotlib.pyplot as plt


def test_batch_duplication():

    t = torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])
    print(t.shape)
    s = batch_duplication(t,5)
    print(s.shape)
    assert t.all == s[0].all
    assert t.shape == s.shape[1:]


def test_grad():
    # create vector field
    t = torch.FloatTensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                           [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                           [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],

                         [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    )

    # create batch of 2 vector fields
    new_shape = list(t.shape)
    new_shape.insert(0, 2)
    new_shape = tuple(new_shape)
    t = torch.cat((t, t)).view(new_shape)
    print(t.shape)

    from metrics import grad as grad
    s = t+1
    assert (grad(t) - grad(s)).bool().all()

def test_sum():
    t = torch.FloatTensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],

                           [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[1, 2, 3], [4, 5, 6], [7, 8, -9]]]]
                          )
    print((t*t).sum())

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
        self.atlas = atlas.to(device)
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
        vector_map = out.permute(0,2,3,4,1)


        warped_image = F.grid_sample(input=original_image, grid=vector_map, padding_mode="reflection")
        return warped_image, vector_map


    @staticmethod
    def single_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False)
        )

if __name__ == "__main__":
    test_sum()