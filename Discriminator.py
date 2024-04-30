import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision  # modules and transforms for computer vision
from matplotlib import pyplot as plt
import tqdm  # progress bar

def discriminator_block(in_filters, out_filters):
    """Return downsampling layers of each discriminator block"""
    layers = [torch.nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
    return layers


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        layers = []
        layers.extend(discriminator_block(in_channels*2, 64))
        layers.extend(discriminator_block(64, 128))
        layers.extend(discriminator_block(128, 256))
        layers.extend(discriminator_block(256, 512))
        layers.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(torch.nn.Conv2d(512, 1, 1, padding=0))
        # layers.append(torch.nn.Conv2d(512, 1, 4, padding=0))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
