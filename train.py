import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import random
import cv2 as cv
import torchvision.transforms as transforms
from data_processing import *
from models.srcnn import SRCNN
from models.edsr import EDSR
from models.rcan import RCAN
from models.utils import *
import pickle
import sys

class checkpoint():
    def __init__(self, name):
        self.name = name
        self.params = {}
        self.loss = []
        self.total_loss = 0
        self.best_psnr = 0
        self.optim_state = None
        self.model_weights = None
        self.iters = 0

    def update_accuracy(self, x, y):
        self.total_y += y.numel()
        self.total_correct += ((x*255).int() == y).sum().item()
        self.acc.append((self.total_correct / self.total_y))

    def update_test_accuracy(self, x, y):
        self.test_total += y.numel()
        self.test_correct += ((x*255).int() == y).sum().item()
        self.test_acc = (self.test_correct / self.test_total)

    def update_loss(self, loss, bidx):
        self.total_loss += loss.item()
        self.loss.append((self.total_loss/(bidx+1)))

    def __repr__(self):
        return f'{self.name} | acc={self.acc[-1]:.4f}'
    
    def load_params(self, model, kwargs):
        for k,v in kwargs.items():
            self.params[k] = v
        for k,v in model.params.items():
            self.params[k] = v

    def reset(self):
        self.total_loss = 0

def train(ckp, model, **kwargs):
    ckp.load_params(model, kwargs)
    model = model.cuda()
    dataset = DIV2k(train=True, size=kwargs['size']) 
    test_dataset = PSNRData('Set5')
    train_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])

    if ckp.model_weights is not None:
        model.load_state_dict(ckp.model_weights)
        optimizer.load_state_dict(ckp.optim_state)
        optimizer.param_groups[0]['lr'] = kwargs['lr']

    for i in range(kwargs['epochs']):
        model.train()
        total_iters = int(dataset.__len__() / kwargs['batch_size'])
        progress_bar = tqdm.tqdm(total=total_iters, desc='Epoch', unit='iter')
        for bidx, (x,y) in enumerate(train_loader):
            ckp.iters += 1
            x = x.cuda()
            y = y.cuda()
            outputs = model(x)
            optimizer.zero_grad()
            loss = criterion(outputs, (y / 255))
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ckp.update_loss(loss, bidx)

            progress_bar.update(1) 
        progress_bar.close()

        ckp.reset()
        model.eval()
        total_psnr = 0
        with torch.no_grad():
            for bidx, (x,y) in enumerate(test_loader):
                x = x.cuda()
                y = y.cuda()
                outputs = model(x)
                x = y_channel(outputs[:,:,8:-8,8:-8] * 255)
                y = y_channel(y[:,:,8:-8,8:-8])
                p = psnr(x, y)
                total_psnr += p
        current_psnr = total_psnr / test_dataset.__len__()
        print(f'Epoch {i+1} | Loss {ckp.loss[-1]:.4f} | PSNR {current_psnr}')
        if current_psnr > ckp.best_psnr:
            ckp.best_psnr = current_psnr
            ckp.model_weights = model.state_dict()
            ckp.optim_state = optimizer.state_dict()
            print("Saving Current Model Weights")
            with open(f'checkpoints/{ckp.name}.pkl', 'wb') as f:
                pickle.dump(ckp, f)

def main():
    #ckp = checkpoint('rcan1')
    with open('checkpoints/rcan1.pkl', 'rb') as f:
        ckp = pickle.load(f)
    model = RCAN()
    train(ckp, model, size=96, lr=10e-5, batch_size=16, epochs=100)

if __name__ == '__main__':
    main()
