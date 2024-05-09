import torch
from torch import nn
from loss_zoo import Normalization, ContentLoss, gram_matrix

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

content_layers_default = ["conv_4"]
style_layers_fefault = ["conv_1",
                        "conv_2",
                        "conv_3",
                        "conv_4",
                        "conv_5"]


def get_style_model_and_losses(norm_mean, norm_std,
                               vgg_model,
                               content_layers, style_layers,
                               content_img, style_img
                               ):

    norm_layer = Normalization(norm_mean, norm_std)

    content_losses = []
    style_losses = []

    model = nn.Sequential(norm_layer)

    i = 0

    for layer in vgg_model.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=True)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"max_pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"batch_norm_{i}"
        else:
            RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

