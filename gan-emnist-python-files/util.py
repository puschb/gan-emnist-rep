import math
import torch
import torch.nn.functional
from tqdm import tqdm
import numpy as np
from numpy.random import default_rng
import matplotlib
from matplotlib import pyplot as plt
import torch.utils.data
import time


def display_generated_images(generator,latent_size):

    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        device = "cpu"

    generator.to(device)
    generator.eval()

    generated_im = generator(torch.tensor(np.random.normal(0,1,(16,latent_size)), dtype=torch.float32, device=device))

    fig = plt.figure(1, figsize=(10, 7))
    rows, columns, count = 4, 4, 1
    with torch.no_grad():
        for im in generated_im:
            fig.add_subplot(rows, columns, count)
            plt.imshow(np.array(im.cpu()).squeeze())
            count +=1

        plt.show()
    generator.train()

#function copied from dcgan source code - I kind of understand how this works but not really --> spend more time trying to understand it
def initialize_weights(model): #get some kind of choosing mechanism in here
    layer_type = model.__class__.__name__
    if layer_type.find("Conv") != -1:
        torch.nn.init.normal_(model.weight.data, 0, 0.02)
    elif layer_type.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(model.weight.data, 0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0)
