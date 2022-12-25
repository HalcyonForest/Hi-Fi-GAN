import torch
from torch import nn
import model.blocks
from model.blocks import ResBlock
from utils import init_weights
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import compute_pad

#  hu = 512, ku = [16, 16, 4, 4], kr = [3, 7, 11], and Dr = [[1, 1], [3, 1], [5, 1]] × 3].


class Generator(nn.Module):

    def __init__(self, config):
        super(Generator, self).__init__()

        upsample = config.upsample

        k_u = config.k_u
        k_r = config.k_r
        D_r = config.D_r
        
        self.L = len(k_u)
        self.M = len(k_r)
        ups = config.upsample

        dim = config.dim

        self.pre_layer = weight_norm(nn.Conv1d(config.n_mels, dim, 7, 1, 3))

        self.upblocks = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for l in range(len(k_u)):
            in_channels = dim // (2 ** l)
            out_channels = in_channels // 2
            upsample = nn.ConvTranspose1d(in_channels, out_channels, k_u[l], stride=ups[l], padding=(k_u[l] - ups[l])//2)
            self.upblocks.append(upsample)
            
            for m in range(len(k_r)):
                padding = compute_pad(k_r[m], D_r[m][0])
                #   kernel_size, num_channels, padding, dilation_rates
                resblock = ResBlock(k_r[m], out_channels, padding, D_r[m])
                self.resblocks.append(resblock)

        self.post_layer = weight_norm(nn.Conv1d(out_channels, 1, 7, 1, 3)).apply(init_weights)
    
        self.upblocks.apply(init_weights)
        self.post_layer.apply(init_weights)
        self.pre_layer.apply(init_weights)


        self.activation = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()
            

    def forward(self, x):
        x = self.pre_layer(x)
        for l in range(self.L):
            x = self.activation(x)
            x = self.upblocks[l](x)
            resblock_sum = 0.0
            for m in range(self.M):
                resblock_index = l * self.M + m 
                resblock_sum = resblock_sum + self.resblocks[resblock_index](x)
            x = resblock_sum / self.M 
        x = self.activation(x)
        x = self.post_layer(x)
        return self.tanh(x)


    def remove_weight_norm(self):
        remove_weight_norm(self.pre_layer)
        for l in range(self.L):
            remove_weight_norm(self.upblocks[l])

        for resblock in self.resblocks:
            resblock.remove_weight_norm()
        
        remove_weight_norm(self.post_layer)
