import torch
import torch.nn as nn
import torch.nn.functional as F


# for full-connected CNN
class up_conv_as_pool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv_as_pool, self).__init__()

        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3))
        self.c2 = nn.Conv2d(in_ch, out_ch, kernel_size=(2, 3))
        self.c3 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 2))
        self.c4 = nn.Conv2d(in_ch, out_ch, kernel_size=(2, 2))

    def forward(self, x):
        x1 = self.c1(F.pad(x, (1, 1, 1, 1)))
        x2 = self.c2(F.pad(x, (1, 1, 0, 1)))
        x3 = self.c3(F.pad(x, (0, 1, 1, 1)))
        x4 = self.c4(F.pad(x, (0, 1, 0, 1)))
        b, c, h, w = x1.shape

        x12 = torch.stack((x1, x2), dim=-3).permute(0, 1, 3, 4, 2).reshape(b, -1, h, w * 2)
        x34 = torch.stack((x3, x4), dim=-3).permute(0, 1, 3, 4, 2).reshape(b, -1, h, w * 2)
        x1234 = torch.stack((x12, x34), dim=-3).permute(0, 1, 3, 2, 4).reshape(b, -1, h * 2, w * 2)

        return x1234
        

class up_proj_as_pool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_proj_as_pool, self).__init__()

        self.a1 = up_conv_as_pool(in_ch, out_ch)
        self.a2 = nn.ReLU(inplace=False)
        self.a3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)

        self.b1 = up_conv_as_pool(in_ch, out_ch)

        self.ab1 = nn.ReLU(inplace=False)

    def forward(self, x):
        a = self.a1(x)
        a = self.a2(a)
        a = self.a3(a)

        b = self.b1(x)

        ab = a + b
        ab = self.ab1(ab)
        return ab


class projection(nn.Module):
    def __init__(self, d0: int, d1: int, d2: int, stride: int):
        super(projection, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(d0, d1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(d1),
            nn.ReLU(inplace=False),
            nn.Conv2d(d1, d1, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(d1),
            nn.ReLU(inplace=False),
            nn.Conv2d(d1, d2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(d2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(d0, d2, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(d2)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = x1 + x2
        return x3


class skip(nn.Module):
    def __init__(self, d1: int, d2: int):
        super(skip, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(d2, d1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(d1),
            nn.ReLU(inplace=False),
            nn.Conv2d(d1, d1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d1),
            nn.ReLU(inplace=False),
            nn.Conv2d(d1, d2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(d2)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x + x1
        return x2


# for U-net
class DoubleConv(nn.Module):
    def __init__(self, d0: int, d1: int, d2: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d0, d1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d1),
            nn.ReLU(inplace=False),
            nn.Conv2d(d1, d2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d2),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x