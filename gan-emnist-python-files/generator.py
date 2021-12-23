import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,im_size = 32, latent_size = 100, num_channels = 3, num_conv_layers = 3, init_num_features = 128): #probably more to add here
        super(Generator,self).__init__()

        self.init_size = im_size // (2**num_conv_layers) #image dimensions have to be a power of 2 and 2**(convlayers+1) can't be greater than im_size
        self.init_num_features = init_num_features

        self.layer1 = nn.Sequential(nn.Linear(latent_size, self.init_size ** 2 * self.init_num_features))

        kernal = 3
        padding = 1 #I will have to change the padding depending on the kernal size

        conv_block = [nn.BatchNorm2d(init_num_features)]
        for i in range(num_conv_layers):
            conv_block.append(nn.Upsample(scale_factor=2))

            if not i == num_conv_layers-1:
                conv_block.append(nn.Conv2d(init_num_features // (2 ** i), init_num_features // (2 ** (i + 1)),
                                            (kernal, kernal), padding=padding))
                conv_block.append(nn.BatchNorm2d(init_num_features//(2**(i+1))))
                conv_block.append(nn.ReLU())

            else:

                conv_block.append(nn.Conv2d(init_num_features // (2 ** i), num_channels, (kernal,kernal),
                                            padding=padding))
                conv_block.append(nn.Tanh())

        self.conv = nn.Sequential(*conv_block)


    def forward(self,z):
        z = self.layer1(z)
        z = z.view(z.shape[0],self.init_num_features,self.init_size,self.init_size)
        z = self.conv(z)
        return z
