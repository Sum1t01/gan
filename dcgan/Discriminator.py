import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, feature_d):
        super().__init__()
        #Input: N * channels_img * 64 * 64
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img, feature_d, kernel_size=4, padding=1, stride=2
                ), #32 * 32
            nn.LeakyReLU(0.2),
            self._block(feature_d, feature_d*2, kernel_size=4, padding=1, stride=2), #16 * 16
            self._block(feature_d*2, feature_d*4, kernel_size=4, padding=1, stride=2), #8 * 8
            self._block(feature_d*4, feature_d*8, kernel_size=4, padding=1, stride=2), #4 * 4
            self._block(feature_d*8, 1, kernel_size=4, padding=0, stride=2), #1 * 1
            nn.Sigmoid()
        ) 

    def _block(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, input):
        return self.disc(input)