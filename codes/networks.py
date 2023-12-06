import torch
import torch.nn as nn
import torch.nn.functional as F

import codes.layers as layer


class fcrn_v1_tiny(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            layer.projection(64, 32, 128, 2),
            nn.ReLU(inplace=False),
            layer.skip(32, 128),
            nn.ReLU(inplace=False),
            layer.skip(32, 128),
            nn.ReLU(inplace=False),
            layer.skip(32, 128),
            nn.ReLU(inplace=False),
            layer.up_proj_as_pool(128, 64),
            nn.ReLU(inplace=False),
            layer.up_proj_as_pool(64, 32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
            

class fcrn_v1_tiny_sub(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            layer.projection(64, 32, 128, 2),
            nn.ReLU(inplace=False),
            layer.skip(32, 128),
            nn.ReLU(inplace=False),
            layer.skip(32, 128),
            nn.ReLU(inplace=False),
            layer.skip(32, 128),
            nn.ReLU(inplace=False),
            layer.up_proj_as_pool(128, 64),
            nn.ReLU(inplace=False),
            layer.up_proj_as_pool(64, 32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class fcrn_v1_small(nn.Module):
    def __init__(self):
        super().__init__()

        self.data = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False)
        ) # 160 -> 80

        self.down = nn.Sequential(
            layer.projection(64, 64, 256, 2),
            nn.ReLU(inplace=False),
            layer.skip(64, 256),
            nn.ReLU(inplace=False),
            layer.skip(64, 256),
            nn.ReLU(inplace=False),
            layer.skip(64, 256),
            nn.ReLU(inplace=False),
            layer.projection(256, 128, 512, 2),
            nn.ReLU(inplace=False),
            layer.skip(128, 512),
            nn.ReLU(inplace=False),
            layer.skip(128, 512),
            nn.ReLU(inplace=False),
            layer.skip(128, 512),
            nn.ReLU(inplace=False),
            layer.skip(128, 512),
            nn.ReLU(inplace=False),
        ) # 80 -> 20

        self.up = nn.Sequential(
            layer.up_proj_as_pool(512, 256),
            nn.ReLU(inplace=False),
            layer.up_proj_as_pool(256, 128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(1)
        ) # 20 -> 80

    def forward(self, x):
        x = self.data(x)
        x = self.down(x)
        x = self.up(x)
        return x


class fcrn_v1_small_sub(nn.Module):
    def __init__(self):
        super().__init__()

        self.data = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False)
        ) # 160 -> 80

        self.down = nn.Sequential(
            layer.projection(64, 64, 256, 2),
            nn.ReLU(inplace=False),
            layer.skip(64, 256),
            nn.ReLU(inplace=False),
            layer.skip(64, 256),
            nn.ReLU(inplace=False),
            layer.skip(64, 256),
            nn.ReLU(inplace=False),
            layer.projection(256, 128, 512, 2),
            nn.ReLU(inplace=False),
            layer.skip(128, 512),
            nn.ReLU(inplace=False),
            layer.skip(128, 512),
            nn.ReLU(inplace=False),
            layer.skip(128, 512),
            nn.ReLU(inplace=False),
            layer.skip(128, 512),
            nn.ReLU(inplace=False),
        ) # 80 -> 20

        self.up = nn.Sequential(
            layer.up_proj_as_pool(512, 256),
            nn.ReLU(inplace=False),
            layer.up_proj_as_pool(256, 128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(1)
        ) # 20 -> 80

    def forward(self, x):
        x = self.data(x)
        x = self.down(x)
        x = self.up(x)
        return x


class cnn_v1(nn.Module):
    def __init__(self):
        super().__init__()

        self.data = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(64),
        ) # 160 -> 80

        self.down = nn.Sequential(
            nn.Conv2d(64, 72, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(72),
            nn.Conv2d(72, 90, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(90),
            nn.Conv2d(90, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 144, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(144),
            nn.Conv2d(144, 160, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(160),
        ) # 80 -> 40

        self.up = nn.Sequential(
            layer.up_conv_as_pool(160, 160),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(160),
            nn.Conv2d(160, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(1)
        ) # 40 -> 80

    def forward(self, x):
        x = self.data(x)
        x = self.down(x)
        x = self.up(x)
        return x


class cnn_v1_sub(nn.Module):
    def __init__(self):
        super().__init__()

        self.data = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(64),
        ) # 160 -> 80

        self.down = nn.Sequential(
            nn.Conv2d(64, 72, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(72),
            nn.Conv2d(72, 90, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(90),
            nn.Conv2d(90, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 144, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(144),
            nn.Conv2d(144, 160, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(160),
        ) # 80 -> 40

        self.up = nn.Sequential(
            layer.up_conv_as_pool(160, 160),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(160),
            nn.Conv2d(160, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(1)
        ) # 40 -> 80

    def forward(self, x):
        x = self.data(x)
        x = self.down(x)
        x = self.up(x)
        return x


class unet_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = layer.DoubleConv(3, 64, 64)
        self.p1 = nn.MaxPool2d(2, 2)
        self.c2 = layer.DoubleConv(64, 128, 128)
        self.p2 = nn.MaxPool2d(2, 2)
        self.c3 = layer.DoubleConv(128, 256, 256)
        self.p3 = nn.MaxPool2d(2, 2)
        self.c4 = layer.DoubleConv(256, 512, 512)
        self.u1 = layer.up_conv_as_pool(512, 256)
        self.u2 = layer.up_conv_as_pool(256, 128)
        self.uc1 = layer.DoubleConv(512, 256, 256)
        self.uc2 = layer.DoubleConv(256, 128, 128)
        self.uc3 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x = self.c1(x)
        x1 = self.p1(x)
        x1 = self.c2(x1)
        x2 = self.p2(x1)
        x2 = self.c3(x2)
        x3 = self.p3(x2)
        x3 = self.c4(x3)

        y2 = self.u1(x3)
        y2 = torch.cat((x2, y2), dim=-3)
        y2 = self.uc1(y2)
        y1 = self.u2(y2)
        y1 = torch.cat((x1, y1), dim=-3)
        y1 = self.uc2(y1)
        y1 = self.uc3(y1)
        return y1
        
        
class unet_v1_sub(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = layer.DoubleConv(4, 64, 64)
        self.p1 = nn.MaxPool2d(2, 2)
        self.c2 = layer.DoubleConv(64, 128, 128)
        self.p2 = nn.MaxPool2d(2, 2)
        self.c3 = layer.DoubleConv(128, 256, 256)
        self.p3 = nn.MaxPool2d(2, 2)
        self.c4 = layer.DoubleConv(256, 512, 512)
        self.u1 = layer.up_conv_as_pool(512, 256)
        self.u2 = layer.up_conv_as_pool(256, 128)
        self.uc1 = layer.DoubleConv(512, 256, 256)
        self.uc2 = layer.DoubleConv(256, 128, 128)
        self.uc3 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x = self.c1(x)
        x1 = self.p1(x)
        x1 = self.c2(x1)
        x2 = self.p2(x1)
        x2 = self.c3(x2)
        x3 = self.p3(x2)
        x3 = self.c4(x3)

        y2 = self.u1(x3)
        y2 = torch.cat((x2, y2), dim=-3)
        y2 = self.uc1(y2)
        y1 = self.u2(y2)
        y1 = torch.cat((x1, y1), dim=-3)
        y1 = self.uc2(y1)
        y1 = self.uc3(y1)
        return y1

