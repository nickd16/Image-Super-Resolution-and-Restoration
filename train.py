import torch 
import torch.nn as nn
from einops import rearrange
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import random
import cv2 as cv
import torchvision.transforms as transforms
from data_processing import DIV2k, PSNRData
from model import EDSR
import pickle
from torchmetrics.image import PeakSignalNoiseRatio

class checkpoint():
    def __init__(self, name):
        self.name = name
        self.loss = []
        self.acc = []
        self.total_loss = 0
        self.total_y = 0
        self.total_correct = 0
        self.test_total = 0
        self.test_correct = 0
        self.optim_state = None
        self.model_weights = None
        self.test_acc = 0
        self.iters = 0

    def update_accuracy(self, x, y):
        self.total_y += y.numel()
        self.total_correct += (x.int() == y).sum().item()
        self.acc.append((self.total_correct / self.total_y))

    def update_test_accuracy(self, x, y):
        self.test_total += y.numel()
        self.test_correct += (x.int() == y).sum().item()
        self.test_acc = (self.total_correct / self.total_y)

    def update_loss(self, loss, bidx):
        self.total_loss += loss.item()
        self.loss.append((self.total_loss/(bidx+1)))

    def __repr__(self):
        return f'{self.name} | acc={self.acc[-1]:.4f}'
    
    def reset(self):
        self.total_loss = 0
        self.test_total = 0
        self.test_correct = 0
        self.total_y = 0
        self.total_correct = 0


def train(ckp, batch_size=16, size=96, lr=10e-4, epochs=1):
    dataset = DIV2k(train=True, size=size) 
    test_dataset = DIV2k(train=False, size=size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    model = EDSR().cuda()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if ckp.model_weights is not None:
        model.load_state_dict(ckp.model_weights)
        optimizer.load_state_dict(ckp.optim_state)

    epochs = epochs
    for i in range(epochs):
        model.train()
        total_iters = int(dataset.__len__() / batch_size)
        progress_bar = tqdm.tqdm(total=total_iters, desc='Epoch', unit='iter')
        for bidx, (x,y) in enumerate(train_loader):
            ckp.iters += 1
            x = x.cuda()
            y = y.cuda()
            outputs = model(x)
            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ckp.update_loss(loss, bidx)
            ckp.update_accuracy(outputs, y)
            progress_bar.update(1) 
            if (bidx+1) % 50 == 0:
                print(f'Epoch {i+1} | Loss {ckp.loss[-1]:.4f} | Accuracy {ckp.acc[-1]:.4f}')
        progress_bar.close()

        model.eval()
        prev_test_acc = ckp.test_acc
        for bidx, (x,y) in enumerate(test_loader):
            x = x.cuda()
            y = y.cuda()
            outputs = model(x)
            ckp.update_test_accuracy(outputs, y)
        print(f'Test Accuray {ckp.test_acc}')
        ckp.reset()
        if ckp.test_acc > prev_test_acc:
            ckp.model_weights = model.state_dict()
            ckp.optim_state = optimizer.state_dict()
            print("Saving Current Model Weights")
            with open(f'checkpoints/{ckp.name}.pkl', 'wb') as f:
                pickle.dump(ckp, f)
        else:
            ckp.test_acc = prev_test_acc

def visual_test(ckp):
    dataset = DIV2k(train=False, size=224, full=True) 
    model = EDSR().cuda().eval()
    if ckp.model_weights is not None:
        model.load_state_dict(ckp.model_weights)
    for _ in range(16):
        r = random.randint(1,99)
        x,y = dataset[r]
        x, y = x.cuda().unsqueeze(0), y.unsqueeze(0)
        output = model(x)
        output = output.cpu()
        x = x.cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        x = ((x*std) + mean) * 255
        display(x,output,False)

def test_pnr(ckp, testset):
    dataset = PSNRData(testset)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = EDSR().eval()
    total_psnr = 0
    psnr = PeakSignalNoiseRatio()
    if ckp.model_weights is not None:
        model.load_state_dict(ckp.model_weights)
    with torch.no_grad():
        for bidx,(x,y) in enumerate(test_loader):
            outputs = model(x)
            p = psnr(outputs, y)
            total_psnr += p
    print(f'Final PSNR {(total_psnr / dataset.__len__()).item()}')

def display(image, image_real, normalize=True):
    image_real = image_real.squeeze(0)
    image = image.squeeze(0)
    image = image.permute(1,2,0)
    image_real = image_real.permute(1,2,0)
    image = image.int().numpy()
    image_real = image_real.int().numpy()
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[1].imshow(image_real)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    #ckp = checkpoint("test1")
    with open('checkpoints/test1.pkl', 'rb') as f:
        ckp = pickle.load(f)
    #train(ckp, batch_size=16, size=96, lr=10e-4, epochs=5)
    #test_pnr(ckp, 'Set14')
    visual_test(ckp)

if __name__ == '__main__':
    main()
