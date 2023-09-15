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
    def __init__(self, train=True):
        path = '../datasets/DIV2K_'
        path += 'train_HR' if train else 'valid_HR'
        self.images = []
        for file in os.listdir(path):
            img = cv.imread(path+'/'+file)
            self.images.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop = transforms.RandomCrop((112,112))
        self.blur = transforms.GaussianBlur(kernel_size=(5,5), sigma=(.1,2.0))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)
        img = self.crop(img)
        x = img.clone().detach()
        x = self.blur(x)
        x = F.interpolate(x.unsqueeze(0), size=(56,56), mode='bicubic').squeeze(0)
        y = img.clone().detach()
        x = x / 255.0
        x = self.normalize(x)
        return (x,y)
    
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
    dataset = DIV2k(train=False)
    x, y = dataset[5]
    display(x,y)

if __name__ == '__main__':
    main()