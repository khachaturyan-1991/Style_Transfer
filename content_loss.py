from __future__ import print_function, division
from builtins import range, input

from datetime import datetime

from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.utils as image

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b

def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img

def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x

def VGG16_AvgPool(shape):
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)
    
    i = vgg.input
    x = i
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            x = AveragePooling2D()(x)
        else:
            x = layer(x)
    
    return Model(i, x)

# def VGG16_AvgPool_CutOff(shape, num_convs):
    
#     model = VGG16_AvgPool(shape)
#     n = 0
#     output = None
#     for layer in model.layers:
#         if layer.__class__ == Conv2D:
#             n += 1
#         if n >= num_convs:
#             output = layer.output
#             break
    
#     return Model(model.input, output)

def VGG16_AvgPool_CutOff(shape, num_convs):
    
    model = VGG16_AvgPool(shape)
    new_model = Sequential()
    new_model.add(Input(shape = shape))
    n=0
    for layer in model.layers:
        if layer.__class__ == Conv2D:
            n += 1
        new_model.add(layer)
        if n >= num_convs:
            break
        
    return new_model