import torch
import matplotlib.pylab as plt
from loss_zoo import Loss
from train_zoo import Train
from utils import image_loader
from model_zoo import extract_features, rebuil_vgg, ImgageBuilder


if __name__ == "__main__":

    vgg_up = rebuil_vgg()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    style_img = image_loader("./imgs/Picasso_1.jpeg").to(device)
    content_img = image_loader("./imgs/Mauntains.jpg").to(device)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    contet_layer_names = "conv_4".split()
    content_features = extract_features(model=vgg_up,
                                        layers_to_extract=contet_layer_names,
                                        img=content_img)
    style_layer_names = "conv_2 conv_3 conv_4 conv_5".split()
    style_features = extract_features(model=vgg_up,
                                      layers_to_extract=style_layer_names,
                                      img=style_img)

    loss_fn = Loss(content_features=content_features,
                   style_features=style_features,
                   model=vgg_up,
                   style_influence=0.995)

    image_generator = ImgageBuilder(image=content_img)

    optimizer = torch.optim.Adam(image_generator.parameters(), lr=1e-3)

    train = Train(generative_model=image_generator,
                  num_of_epochs=3e2,
                  loss_function=loss_fn,
                  optimizer=optimizer)

    loss_history = train.train_step()

    plt.plot([i.item() for i in loss_history])
