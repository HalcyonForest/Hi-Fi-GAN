import torch 
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm
from utils import compute_pad
from utils import init_weights

#  hu = 512, ku = [16, 16, 4, 4], kr = [3, 7, 11], and Dr = [[1, 1], [3, 1], [5, 1]] × 3].
class ResBlock(nn.Module):
    def __init__(self, kernel_size, num_channels, padding, dilation_rates = [1, 3, 5]):
        super(ResBlock, self).__init__()
        
        self.net = nn.ModuleList()
        self.activation = nn.LeakyReLU(0.1)

        for d in dilation_rates:
            current_padding = compute_pad(kernel_size, d)
            current_padding_1 = compute_pad(kernel_size, 1)
#           print("Paddings: ", current_padding, current_padding_1)
            self.net.append(
                weight_norm(
                  nn.Conv1d(num_channels, num_channels, kernel_size, 1, dilation=d,
                               padding=current_padding)
                )
            )
            self.net.append(
                weight_norm(
                    nn.Conv1d(num_channels, num_channels, kernel_size, 1, dilation=1,
                               padding=current_padding_1)
                )
          )
    
        self.net.apply(init_weights)
#         print(self.net)



        
    def forward(self, x):
        for i in range(0, len(self.net), 2):
            res = x
            x = self.activation(x)
            x = self.net[i](x)
            x = self.activation(x)
            x = self.net[i+1](x)
            x = x + res
        return x


    def remove_weight_norm(self):
        for l in self.net:
            remove_weight_norm(l)
        
        

