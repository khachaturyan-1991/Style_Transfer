import numpy as np
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
IMG_SIZE = 128

img_transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                    transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
                                    transforms.ToTensor()])
img_untransform = transforms.ToPILImage()


def load_img(img_path: str,
             device: str):
    img = Image.open(img_path)
    img = img_transform(img)
    img = img.to(device, torch.float)
    return img


class Normalization(nn.Module):

    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img-self.mean)/self.std


def imshow(img, title=None):
    img = torch.tensor.cpu().clone()
    img = img.squeeze(0)
    img = img_untransform(img)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)
    plt.show()


if __name__ == "__main__":
    device = "mps"
    style_img = load_img("./imgs/VanGoh.jpg", device=device)
    content_img = load_img("./imgs/Angelica.png", device=device)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    print(style_img.shape)
