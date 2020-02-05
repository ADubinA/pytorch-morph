import torch.nn as nn


class AutoEncoderV1(nn.Module):

    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.maxpool = nn.MaxPool3d(2)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            self.maxpool,

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            self.maxpool,

            # nn.Conv3d(32, 64, kernel_size=3, padding=1),
            # nn.ReLU(True),
            # self.maxpool,
        )
        self.decoder = nn.Sequential(
            # self.upsample,
            # nn.Conv3d(64, 32, kernel_size=3, padding=1),
            # nn.ReLU(True),

            self.upsample,
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(True),

            self.upsample,
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
        )



    @staticmethod
    def single_conv(in_channels, out_channels, activation="relu"):
        if activation == "relu":
            act = nn.ReLU(True)
        else:
            act = nn.Tanh()


        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            act,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            act,
            )

    def forward(self, volume):

        latent = self.encoder(volume)
        x = self.decoder(latent)

        return x



