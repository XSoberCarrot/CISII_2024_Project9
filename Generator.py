import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision  # modules and transforms for computer vision
from matplotlib import pyplot as plt
import tqdm  # progress bar

class ResUNet(torch.nn.Module):
    """Custom U-Net architecture"""

    def __init__(self, in_channels=1, out_channels=1, p=0.25):
        """Initializes U-Net."""

        super(ResUNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(48, 48, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = torch.nn.Sequential(
            torch.nn.Conv2d(48, 48, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = torch.nn.Sequential(
            torch.nn.Conv2d(48, 48, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 96, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(96, 96, 3, stride=1, padding=1),
            torch.nn.Dropout(0.25),
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = torch.nn.Sequential(
            torch.nn.Conv2d(144, 96, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(96, 96, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = torch.nn.Sequential(
            torch.nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            #torch.nn.Conv2d(96, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 32, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, out_channels, 3, stride=1, padding=1))

        self._leakyAct = torch.nn.LeakyReLU(0.1)
        self.drop_layer = torch.nn.Dropout(p)

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool2 = self.drop_layer(pool2)
        pool3 = self._block2(pool2)
        pool3 = self.drop_layer(pool3)
        pool4 = self._block2(pool3)
        pool4 = self.drop_layer(pool4)
        pool5 = self._block2(pool4)
        pool5 = self.drop_layer(pool5)

        # Decoder
        upsample5 = self._block3(pool5)
        upsample5 = self.drop_layer(upsample5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        upsample4 = self.drop_layer(upsample4)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        upsample3 = self.drop_layer(upsample3)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        upsample2 = self.drop_layer(upsample2)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        res = self._block6(concat1)
        im = self._leakyAct(res)

        # Final activation
        return im
