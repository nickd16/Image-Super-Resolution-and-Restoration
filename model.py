import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.conv1 = nn.Conv2d(F, F, 3, 1, 1)
        self.conv2 = nn.Conv2d(F, F, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + (self.conv2(self.relu(self.conv1(x))) * 0.1)

class ResnetBackbone(nn.Module):
    def __init__(self, B, F):
        super().__init__()
        self.B = B
        for i in range(B):
            setattr(self, f'resblock{i+1}', ResBlock(F))

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
    def __init__(self, B=32, F=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, F, 3, 1, 1)
        self.backbone = ResnetBackbone(B=B, F=F)
        self.conv2 = nn.Conv2d(F, F, 3, 1, 1)
        self.upsample = Upsample(F)
        self.conv3 = nn.Conv2d(int(F/4), 3, 1)
    
    def forward(self, x):
        res = self.conv1(x)
        x = res + self.conv2(self.backbone(res))
        return self.conv3(self.upsample(x)) * 255
    
def main():
    model = EDSR().cuda()
    x = torch.randn((8,3,112,112)).cuda()
    print(model(x).shape)
    

if __name__ == '__main__':
    main()