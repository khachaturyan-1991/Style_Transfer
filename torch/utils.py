import torch
from torchvision import transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
imsize = 128

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()])


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
