import matplotlib.pyplot as plt
import numpy as np
import torch.nn
from generator import Generator
from discriminator import Discriminator
import training as t
import util as u
import torchvision

IMAGE_SIZE = 32
LATENT_SIZE = 100
NUM_CHANNELS = 1
NUM_LAYERS_G = 5
NUM_LAYERS_D = 5
INITIAL_NUMBER_OF_FEATURES_G = 1024
INITIAL_NUMBER_OF_FEATURES_D = 64
EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 0.001
ADAM_B1 = 0.5
COST_FUNCTION = torch.nn.BCELoss() #this is the cost function from the DCGAN paper, maybe check if there are others too that are better
DATASET_PATH = "C:/Users/Benjamin/Documents/Python/Pytorch_MNIST_Data"

transformations = torchvision.transforms.Compose([ torchvision.transforms.Resize(IMAGE_SIZE),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(.5,.5,)
                                             ])

emnist_blanced_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(DATASET_PATH + '/.mnistData',
                                                    train=True, download=True, transform=transformations), batch_size = BATCH_SIZE,
                                                    shuffle = True)

'''data = np.array(list(emnist_blanced_loader)[0][0])
for i in range(20):
    plt.imshow(data[i].squeeze())
    plt.show()'''
generator = Generator(num_channels=NUM_CHANNELS)
discriminator = Discriminator(num_channels=NUM_CHANNELS)

generator.apply(u.initialize_weights)
discriminator.apply(u.initialize_weights)

g_optimizer = torch.optim.Adam(generator.parameters(),LEARNING_RATE,betas=(ADAM_B1,.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(),LEARNING_RATE, betas=(ADAM_B1,.999))


t.train(generator,discriminator,g_optimizer,d_optimizer,emnist_blanced_loader,EPOCHS,COST_FUNCTION,LATENT_SIZE)

u.display_generated_images(generator,LATENT_SIZE)