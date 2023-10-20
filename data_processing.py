import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import random

class DIV2k(Dataset):
    def __init__(self, size=224, kernel_size=(5,5), sigma=(.1,2), train=True, full=False):
        self.size = size
        path = '../datasets/DIV2K_'
        path += 'train_HR' if train else 'valid_HR'
        self.images = []
        for file in os.listdir(path):
            img = cv.imread(path+'/'+file)
            if full:
                img = cv.resize(img, (size, size))
            self.images.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        self.normalize = transforms.Normalize(mean=[0.4291, 0.4081, 0.3290], std=[0.2513, 0.2327, 0.2084])
        self.crop = transforms.RandomCrop((size,size))
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)
        img = self.crop(img)
        x = img.clone().detach()
        x = self.blur(x)
        x = F.interpolate(x.unsqueeze(0), size=(int(x.shape[1]/2),int(x.shape[2]/2)), mode='bicubic').squeeze(0)
        y = img.clone().detach()
        x = x / 255.0
        x = self.normalize(x)
        return (x,y)
    
class PSNRData(Dataset):
    def __init__(self, dataset):
        path = f'../datasets/data/{dataset}/image_SRF_2'
        self.lr_images = []
        self.hr_images = []
        for file in os.listdir(path):
            img = cv.imread(f'{path}/{file}')
            if 'LR' in file:
                self.lr_images.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            else:
                self.hr_images.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        self.normalize = transforms.Normalize(mean=[0.4291, 0.4081, 0.3290], std=[0.2513, 0.2327, 0.2084])

    def __len__(self):
        return len(self.lr_images)
    
    def __getitem__(self, index):
        lr = torch.tensor(self.lr_images[index], dtype=torch.float32).permute(2, 0, 1)
        hr = torch.tensor(self.hr_images[index], dtype=torch.float32).permute(2, 0, 1)
        lr = lr / 255.0
        lr = self.normalize(lr)
        return (lr,hr)

def display(image, image_real):
    image = image.squeeze(0).permute(1,2,0).int().numpy()
    image_real = image_real.squeeze(0).permute(1,2,0).int().numpy()
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[1].imshow(image_real)
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

def main():
    dataset = DIV2k(train=True, size=96)
    mean = torch.zeros(3)
    # std = torch.zeros(3)
    # for _ in range(dataset.__len__()):
    #     x = dataset[0]
    #     x = x / 255.0
    #     std += torch.std(x, dim=(1,2))
    #     mean += torch.mean(x, dim=(1,2))
    # print(mean / dataset.__len__())
    # print(std / dataset.__len__())

if __name__ == '__main__':
    main()