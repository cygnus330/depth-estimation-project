import os
from os import path
import sys
from glob import glob
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, random_split
import cv2
import pandas as pd


class nyu_v2_kaggle(Dataset):
    def __init__(
        self,
        dir: str,
        resizer: int = 1,
        x_res = (120, 160),
        y_res = (120, 160),
        subset = None
):
        self.label = pd.read_csv(path.join(dir), header=None, names=['x', 'y'])
        self.len = len(self.label)
        
        self.resizer = resizer
        self.subset = subset
        self.x_res = torchvision.transforms.Resize(size=x_res)
        self.y_res = torchvision.transforms.Resize(size=y_res)

    def read_image(self, dir, mode):
        img = cv2.imread(dir, mode)
        if mode == 0 or mode == cv2.IMREAD_GRAYSCALE:
            img = img[:, :, np.newaxis]

        if img.dtype == np.uint16:
            img = img / (2 ** 8)
            img = img.astype(np.uint8)

        return img

    def to_tensor(self, arr):
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        arr = torch.from_numpy(arr)
        return arr

    def __len__(self):
        return self.len // self.resizer

    def __getitem__(self, idx):
        x, y = self.label.iloc[idx]
        x, y = self.read_image(x, 1), self.read_image(y, 0)

        if self.subset:
            x1 = self.subset(x)
            x1 = x1[:, :, np.newaxis]
            x1 = self.to_tensor(x1)

        x, y = self.to_tensor(x), self.to_tensor(y)

        if self.subset:
            x = torch.cat((x, x1), dim=-3)

        x, y = self.x_res(x), self.y_res(y)
            
        x = x / 255
        y = y / 255
        return x, y