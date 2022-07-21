import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class IceBergDataset(Dataset):

    def __init__(self, image, label, transform=None, phase='train'):
        self.image = image
        self.label = label
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        if self.transform:
            img_transformed = self.transform(self.image[index], self.phase)
        else:
            img_transformed = self.image[index]
        label = self.label[index]

        return img_transformed, label