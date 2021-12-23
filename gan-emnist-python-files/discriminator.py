import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, im_size = 32, num_layers = 4, num_channels = 3, init_num_features = 16):
        super(Discriminator,self).__init__()

        final_featureMap_area = (im_size//2**num_layers)**2
        final_num_features = init_num_features * 2**(num_layers-1)
        self.out_layer = nn.Sequential(nn.Linear( final_num_features * final_featureMap_area, 1), nn.Sigmoid())

        kernal = 3
        padding = 1
        stride = 2

        conv_block = []
        for i in range(num_layers):
            if i ==0:
                conv_block.append(nn.Conv2d(num_channels, init_num_features * (2 ** i), (kernal, kernal), stride=stride,
                    padding=padding))
            else:
                conv_block.append(nn.Conv2d(init_num_features*(2**(i-1)), init_num_features*(2**i), (kernal,kernal),
                                            stride = stride, padding = padding))
            conv_block.append(nn.LeakyReLU(.2))
            conv_block.append(nn.Dropout2d(.25)) #this was included in the source code but not in the journal
            if i>0:
                conv_block.append(nn.BatchNorm2d(init_num_features*(2**i)))

        self.conv = nn.Sequential(*conv_block)
        print(self.conv)
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.shape[0],-1)
        x = self.out_layer(x)

        return x