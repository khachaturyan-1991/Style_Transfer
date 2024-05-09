import torch
import torch.nn as nn


class Loss(nn.Module):

    def __init__(self,
                 model,
                 content_features,
                 style_features,
                 style_influence=0.8) -> None:
        super().__init__()
        self.model = model
        self.content_features = content_features
        self.style_features = {key: 0 for key in style_features.keys()}
        for key in self.style_features:
            y = style_features[key].detach().requires_grad_(False)
            self.style_features[key] = self._gram_matrix(y)
        self.cont_layers = content_features.keys()
        self.style_layers = style_features.keys()
        self.coeff = style_influence

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def _extract_features(self, x):
        content_output, style_output = {}, {}
        for name, layer in self.model.named_children():
            x = layer(x)
            if name in self.cont_layers:
                content_output[name] = x
            elif name in self.style_layers:
                style_output[name] = x
        return content_output, style_output

    def forward(self, x):
        loss = 0
        content_part, style_part = self._extract_features(x)
        for key in content_part.keys():
            y = self.content_features[key]
            x = content_part[key]
            loss += (1-self.coeff)*torch.nn.functional.mse_loss(x, y)
        for key in style_part.keys():
            y = self.style_features[key]
            x = self._gram_matrix(style_part[key])
            loss += self.coeff*torch.nn.functional.mse_loss(x, y)
        return loss


if __name__ == "__main__":

    from model_zoo import ImgageBuilder, extract_features, rebuil_vgg
    from utils import image_loader
    from torchvision.models import vgg19, VGG19_Weights

    style_img = image_loader("./imgs/Saryan_mountains.jpeg")
    content_img = image_loader("./imgs/me.jpg")

    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    vgg_up = rebuil_vgg(vgg)
    content_features = extract_features(model=vgg_up,
                                        layers_to_extract=["conv_4"],
                                        img=content_img)
    style_features = extract_features(model=vgg_up,
                                      layers_to_extract=['conv_3', 'conv_5'],
                                      img=style_img)

    image_generator = ImgageBuilder(image=content_img)

    loss_fn = Loss(content_features=content_features,
                   style_features=style_features,
                   model=vgg_up,
                   style_influence=0.5)

    image_generator = ImgageBuilder(image=content_img)

    first_generated_img = image_generator()
    first_generated_img.requires_grad = False
    loss = loss_fn(first_generated_img)
    print("Calculated Loss: ", loss)
    loss.backward()
