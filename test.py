import random
import torch
import pickle
from train import checkpoint
import matplotlib.pyplot as plt
from data_processing import *
from models.utils import *
from models.srcnn import SRCNN
from models.edsr import EDSR
from models.rcan import RCAN

def compare(checkpoints, metric='loss'):
    ckps = []
    for ckp in checkpoints:
        with open(f'checkpoints/{ckp}.pkl', 'rb') as f:
            ckps.append(pickle.load(f))
    for i in range(len(ckps)):
        plt.plot(range(1, len(getattr(ckps[i], f'{metric}'))+1), getattr(ckps[i], f'{metric}'), label=ckps[i].name)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def visual_test(model, ckp):
    dataset = DIV2k(train=False, size=224, sigma=(1,7), full=True) 
    model = model.cuda().eval()
    if ckp.model_weights is not None:
        model.load_state_dict(ckp.model_weights)
    for _ in range(16):
        r = random.randint(1,99)
        x,y = dataset[r]
        x, y = x.cuda().unsqueeze(0), y.unsqueeze(0)
        output = model(x).cpu()
        test_img = (y / 255) - output
        test_img = test_img.squeeze(0).permute(1,2,0).detach().numpy()
        plt.imshow(test_img, cmap='coolwarm')
        output = output * 255
        output = output.cpu()
        x = (simple_meanshift(x) * 255).cpu()
        display(output,y,False)
        #display(output, y, False)

def test_pnr(model, ckp):
    model = model.eval()
    datasets = ['Set5', 'Set14', 'BSD100', 'Urban100']
    for d in datasets:
        dataset = PSNRData(d)
        test_loader = DataLoader(dataset, batch_size=1)
        total_psnr = 0
        if ckp.model_weights is not None:
            model.load_state_dict(ckp.model_weights)
        with torch.no_grad():
            for bidx,(x,y) in enumerate(test_loader):
                outputs = model(x)
                x = y_channel(outputs[:,:,8:-8,8:-8] * 255)
                y = y_channel(y[:,:,8:-8,8:-8])
                p = psnr(x, y)
                total_psnr += p
        print(f'{d} {(total_psnr / dataset.__len__()).item()}')

def image_test(model, ckp, path):
    normalize = transforms.Normalize(mean=[0.4291, 0.4081, 0.3290], std=[0.2513, 0.2327, 0.2084])
    model = model.eval()
    model.load_state_dict(ckp.model_weights)
    img = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
    x = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    x = x / 255
    x = normalize(x)
    img = (model(x) * 255).squeeze(0).int().permute(1,2,0).numpy()
    cv.imwrite('output.png', img)

def main():
    compare(['rcan_test1', 'rcan_test2'], 'PSNR')
    # with open('checkpoints/rcan_test1.pkl', 'rb') as f:
    #     ckp = pickle.load(f)


if __name__ == '__main__':
    main()