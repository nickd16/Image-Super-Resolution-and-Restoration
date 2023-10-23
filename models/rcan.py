import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.utils import *

class ChannelAttention(nn.Module):
    def __init__(self, C, R):
        super().__init__()
        self.adap = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = nn.Conv2d(C, (C//R), 1)
        self.conv2 = nn.Conv2d((C//R), C, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.adap(x)
        s = self.conv2(self.relu(self.conv1(s)))
        s = self.sigmoid(s)
        return (x * s)

class RCAB(nn.Module):
    def __init__(self, C, R):
        super().__init__()
        self.CA = ChannelAttention(C=C, R=R)
        self.conv1 = nn.Conv2d(C, C, 3, 1, 1)
        self.conv2 = nn.Conv2d(C, C, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True) 

    def forward(self, x):
        r = self.conv1(x)
        r = self.relu(r)
        r = self.conv2(r)
        r = self.CA(r)
        return x + r

class ResidualGroup(nn.Module):
    def __init__(self, B, C, R):
        super().__init__()
        self.B = B
        self.conv1 = nn.Conv2d(C, C, 3, 1, 1)
        for i in range(B):
            setattr(self, f'rcab{i+1}', RCAB(C=C, R=R))

    def forward(self, x):
        r = x.clone()
        for i in range(self.B):
            layer = getattr(self, f'rcab{i+1}')
            x = layer(x)
        x = self.conv1(x)
        return x + r
    
class RIR(nn.Module):
    def __init__(self, G, B, C, R):
        super().__init__()
        self.G = G
        self.conv1 = nn.Conv2d(C, C, 3, 1, 1)
        for i in range(G):
            setattr(self, f'resgroup{i+1}', ResidualGroup(B=B, C=C, R=R))

    def forward(self, x):
        r = x.clone()
        for i in range(self.G):
            layer = getattr(self, f'resgroup{i+1}')
            x = layer(x)
        x = self.conv1(x)
        return x + r

class Upsample(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.conv1 = nn.Conv2d(F, F, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuffle(self.conv1(x))
  
class RCAN(nn.Module):
    def __init__(self, G=10, B=20, C=64, R=16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, C, 3, 1, 1)
        self.rir = RIR(G=G, B=B, C=C, R=R)
        self.upsample = Upsample(F=C)
        self.conv2 = nn.Conv2d(int(C/4), 3, 3, 1, 1)
        self.params = {
            'G':G,
            'B':B,
            'C':C,
            'R':R
        }

    def forward(self, x):
        return self.conv2(self.upsample(self.rir(self.conv1(x))))

def main():
    x = torch.randn((16, 3, 48, 48)).cuda()
    model = RCAN().cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)


if __name__ == '__main__':
    main()