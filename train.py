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
from data_processing import DIV2k, PSNRData
from edsr import *
from utils import *
import pickle
import sys

class checkpoint():
    def __init__(self, name):
        self.name = name
        self.params = {}
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
    
    def load_params(self, model, kwargs):
        for k,v in kwargs.items():
            self.params[k] = v
        for k,v in model.params.items():
            self.parmas[k] = v

    def reset(self):
        self.total_loss = 0
        self.test_total = 0
        self.test_correct = 0
        self.total_y = 0
        self.total_correct = 0

def train(ckp, model, **kwargs):
    ckp.load_params(model, kwargs)
    model = model.cuda()
    dataset = DIV2k(train=True, size=kwargs['size']) 
    test_dataset = DIV2k(train=False, size=kwargs['size'])
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
            loss = criterion(outputs, y)
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

def main():
    #ckp = checkpoint('test4')
    with open('checkpoints/test4.pkl', 'rb') as f:
        ckp = pickle.load(f)
    model = EDSR(F=64, B=16, res_scale=0.1)
    train(model, ckp, size=96, lr=10e-5, batch_size=16, epochs=100)
    #test_pnr(ckp, 'Set5', C=64)
    #visual_test(ckp)

if __name__ == '__main__':
    main()
