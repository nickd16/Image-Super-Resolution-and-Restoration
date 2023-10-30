import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.utils import *

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2,2))
    # --------------------layer 1------------------- #
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1_3 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv1_4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upconv5 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.output = nn.Conv2d(64, 3, 1)
    # --------------------layer 2------------------- #
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_3 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, 2)
    # --------------------layer 3------------------- #
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
    # --------------------layer 4------------------- #    
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
    # --------------------layer 5------------------- #   
        self.conv5_1 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2)

        self.params = {}

    def forward(self, x):
        res1 = self.relu(self.conv1_2(self.relu(self.conv1_1(x))))
        res2 = self.relu(self.conv2_2(self.relu(self.conv2_1(self.maxpool(res1)))))
        res3 = self.relu(self.conv3_2(self.relu(self.conv3_1(self.maxpool(res2)))))
        res4 = self.relu(self.conv4_2(self.relu(self.conv4_1(self.maxpool(res3)))))
        x = self.relu(self.conv5_2(self.relu(self.conv5_1(self.maxpool(res4)))))
        x = torch.concat([self.upconv1(x), res4], dim=1)
        x = self.relu(self.conv4_4(self.relu(self.conv4_3(x))))
        x = torch.concat([self.upconv2(x), res3], dim=1)
        x = self.relu(self.conv3_4(self.relu(self.conv3_3(x))))
        x = torch.concat([self.upconv3(x), res2], dim=1)
        x = self.relu(self.conv2_4(self.relu(self.conv2_3(x))))
        x = torch.concat([self.upconv4(x), res1], dim=1)
        x = self.relu(self.conv1_4(self.relu(self.conv1_3(x))))
        x = self.output(self.upconv5(x))
        return simple_meanshift(x)
    
def main():
    x = torch.randn((16, 3, 112, 112)).cuda()
    model = UNET().cuda()
    print(model(x).shape)

if __name__ == '__main__':
    main()    