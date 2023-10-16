import random
import torch
from model import EDSR
import pickle
from train import checkpoint
import matplotlib.pyplot as plt
from data_processing import *
from utils import *

def compare_loss(checkpoints):
    ckps = []
    for ckp in checkpoints:
        with open(f'checkpoints/{ckp}.pkl', 'rb') as f:
            ckps.append(pickle.load(f))
    for i in range(len(ckps)):
        plt.plot(range(1, len(ckps[i].loss)+1), ckps[i].loss, label=ckps[i].name)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

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

def test_pnr(ckp, F=256, B=16, res_scale=0.1):
    model = EDSR(F=F, B=B, res_scale=res_scale).eval()
    datasets = ['Set5', 'Set14', 'BSD100', 'Urban100']
    for d in datasets:
        dataset = PSNRData(d)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        total_psnr = 0
        if ckp.model_weights is not None:
            model.load_state_dict(ckp.model_weights)
        with torch.no_grad():
            for bidx,(x,y) in enumerate(test_loader):
                # x = F.interpolate(x, size=(y.shape[2],y.shape[3]), mode='bicubic')
                # x = torch.clamp(x, 0, 255)
                outputs = model(x)
                x = y_channel(outputs)
                y = y_channel(y)
                p = psnr(x, y)
                total_psnr += p
        print(f'{d} {(total_psnr / dataset.__len__()).item()}')

def main():
    compare_loss(['test1', 'test2', 'test3'])
    # with open('checkpoints/test1.pkl', 'rb') as f:
    #     ckp = pickle.load(f)
    # test_pnr(ckp, F=64, B=16)

if __name__ == '__main__':
    main()