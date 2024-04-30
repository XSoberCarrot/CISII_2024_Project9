# This is file for data set creation
import os
import pathlib
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision  # modules and transforms for computer vision
from matplotlib import pyplot as plt
import tqdm  # progress bar


class Dataset_cGAN(Dataset):
    def __init__(self, database, subset, noiseFunc=None, transform=None):
        """Constructor.

        database: path to the base folder for 2 kind of image folder.
        subset: can be 'train', 'val', or 'test'.
        noiseFunc: function to introduce "noise" perturbations in our dataset, None for identity
        transform: transforms used for data augmentation
        """
        super().__init__()  # good practice to call the base constructor
        # set the folder and get a list of files
        self._dir_GL = pathlib.Path(database) / subset/'GL'
        self._dir_NGL = pathlib.Path(database) / subset/'NGL'
        self.files_GL = sorted(os.listdir(self._dir_GL))
        self.files_NGL = sorted(os.listdir(self._dir_NGL))
        # Define the nose function and transform
        self._noiseFunc = noiseFunc
        self._transform = transform

    def __len__(self):
        """Database size."""
        return len(self.files_NGL)

    def __getitem__(self, index):
        """Image in T1 and its corresponding T2 label."""
        t1_path = self._dir_GL / self.files_GL[index]
        t2_path = self._dir_NGL / self.files_NGL[index]
        image = Image.open(t1_path)
        label = Image.open(t2_path)
        # cast NumPy array to Torch tensor
        image = torchvision.transforms.ToTensor()(image)
        image = torchvision.transforms.Grayscale(num_output_channels=1)(
            image)  # Convert image into grayscale image
        label = torchvision.transforms.ToTensor()(label)
        label = torchvision.transforms.Grayscale(num_output_channels=1)(
            label)  # Convert image into grayscale image
        if self._noiseFunc is not None:
            image = self._noiseFunc(image)
            label = self._noiseFunc(label)
        if self._transform is not None:
            image = self._transform(image)
            label = self._transform(label)
        return image, label

# Test dataset
# Test the data set
# database =  'E:\Data\Work\JHU\MSE\Courses\EN.601.456.01.SP24ComputerIntegratedSurgeryII\Project\DeepLearningCode_GIT\GANS\GLDataset'
# subset = 'train'
# dataset = Dataset_cGAN(database, subset )
#
# print(len(dataset))
# img, label = dataset [1]
# # trans_to_pil = torchvision.transforms.ToPILImage(mode="RGB")
# trans_to_pil = torchvision.transforms.ToPILImage()
# img_pil = trans_to_pil(img)
# img_pil.show()
# print("finish")
