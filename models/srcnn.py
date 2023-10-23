import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.utils import *

class SRCNN(nn.Module):
    def __init__(self, n1=64, n2=32, f1=3, f2=3, f3=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, n1, f1, 1, 1)
        self.conv2 = nn.Conv2d(n1, n2, f2, 1, 1)
        self.conv3 = nn.Conv2d(n2, 3, f3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.params = {
            'n1':n1,
            'n2':n2,
            'f1':f1,
            'f2':f2,
            'f3':f3
        }

    def forward(self, x):
        h, w = x.shape[2:]
        x = F.interpolate(x, size=((h*2),(w*2)), mode='bicubic')
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return simple_meanshift(x)
    
def main():
    x = torch.randn((1, 3, 48, 48)).cuda()
    model = SRCNN().cuda()
    print(model(x).shape)

if __name__ == '__main__':
    main()
