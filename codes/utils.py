import cv2
from os import path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import io

def read_image(dir, mode, res=(120, 160), c=None):
    def to_tensor(arr):
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        arr = torch.from_numpy(arr)
        return arr
    
    conv = torchvision.transforms.Resize(size=res)
    img = cv2.imread(path.join(dir), mode)
    if mode == 0 or mode == cv2.IMREAD_GRAYSCALE:
        img = img[:, :, np.newaxis]
    if img.dtype == np.uint16:
            img = img / (2 ** 8)
            img = img.astype(np.uint8)

    if c is not None:
        img1 = c(img)
        img1 = img1[:, :, np.newaxis]
        img1 = to_tensor(img1)

    img = to_tensor(img)
    if c is not None:
        img = torch.cat((img, img1), dim=-3)
    img = torch.unsqueeze(img, 0)
    img = conv(img)
    img = img / 255
    return img

def make_plot(x1, y1, x2, y2):
    fig = plt.figure(figsize=(12, 16))
    
    plt.subplot(2, 2, 1)
    plt.imshow(x1)
    plt.subplot(2, 2, 2)
    plt.imshow(y1)
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(x2)
    plt.subplot(2, 2, 4)
    plt.imshow(y2)
    plt.colorbar()

    return fig

def plot_to_img(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(fig)
    buf.seek(0)
    file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    image_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 이미지를 PyTorch 텐서로 변환
    transform = transforms.ToTensor()
    tensor_image = transform(image_cv2)

    return tensor_image