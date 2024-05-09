import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights


class Normalization(nn.Module):
    """image normalization"""
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


# normalization params good for vgg
cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]
normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)


def rebuil_vgg():
    """rename VGG layers to easy extrac  Conv2d layers"""
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    vgg_up = nn.Sequential(normalization)
    i = 0
    for layer in vgg.children():
        layer.requires_grad_ = False
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError(
                'Unrecognized layer: {}'.format(layer.__class__.__name__)
                )

        vgg_up.add_module(name, layer)
    return vgg_up


def extract_features(model,
                     layers_to_extract,
                     img):
    """
    Parameters
    -------
    model: nn.Module
        extracts returns of essential layers
    layers_to_extract: list of str
        names of layers which output should be extracted
    img: np.array
        an images that should be passed throw the layers_to_extract
    
    Return
    -------
    extracted_features: list of np.arrays
        list of np.arrays with extracted features
        that should be compared to generate the image
    """
    extracted_features = {}
    for name, layer in model.named_children():
        img = layer(img)
        if name in layers_to_extract:
            extracted_features[name] = \
                img.clone().detach().requires_grad_(False)
    return extracted_features


class ImgageBuilder(nn.Module):
    """
    The nn that generates the image
    self.image is a parameter that is trained
    to match both style and context
    """
    def __init__(self, image) -> None:
        super().__init__()
        self.image = torch.nn.Parameter(torch.rand(image.size()))

    def forward(self):
        return self.image
