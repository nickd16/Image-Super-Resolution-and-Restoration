import torch
import matplotlib.pyplot as plt

def psnr(img1, img2):
    return 20*(torch.log10((255)/(torch.sqrt(torch.mean((img2 - img1) ** 2)))))

def y_channel(x):
    Y = (x.squeeze(0).permute(1, 2, 0)) * (torch.tensor([0.299, 0.587, 0.114]).unsqueeze(0).unsqueeze(0))
    return torch.sum(Y, dim=2)

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