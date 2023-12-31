import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.utils import *

class ResBlock(nn.Module):
    def __init__(self, F, res_scale):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(F, F, 3, 1, 1)
        self.conv2 = nn.Conv2d(F, F, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + (self.conv2(self.relu(self.conv1(x))) * self.res_scale)

class ResnetBackbone(nn.Module):
    def __init__(self, B, F, res_scale):
        super().__init__()
        self.B = B
        for i in range(B):
            setattr(self, f'resblock{i+1}', ResBlock(F, res_scale))

    def forward(self, x):
        for i in range(self.B):
            layer = getattr(self, f'resblock{i+1}')
            x = layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.conv1 = nn.Conv2d(F, F, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuffle(self.conv1(x))

class EDSR(nn.Module):
    def __init__(self, B=32, F=256, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, F, 3, 1, 1)
        self.backbone = ResnetBackbone(B=B, F=F, res_scale=res_scale)
        self.conv2 = nn.Conv2d(F, F, 3, 1, 1)
        self.upsample = Upsample(F)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(int(F/4), 3, 3, 1, 1)
        self.params = {
            'B':B,
            'F':F,
            'res_scale':res_scale 
        }
    
    def forward(self, x):
        res = self.conv1(x)
        x = res + self.conv2(self.backbone(res))
        x = self.conv3(self.relu(self.upsample(x)))
        return simple_meanshift(x)

def main():
    model = EDSR().cuda()
    x = torch.randn((8,3,112,112)).cuda()
    print(model(x).shape)
    
