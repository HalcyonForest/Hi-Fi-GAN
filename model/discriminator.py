import torch
from torch import nn
from torch.nn import functional as F
from model.blocks import ResBlock
from utils import init_weights
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import compute_pad

class DiscriminatorP(nn.Module):
    
    def __init__(self, p, kernel_size=5, stride=3, normalize_spectral=False):
        super(DiscriminatorP, self).__init__()
        self.p = p
        norm = weight_norm
        if normalize_spectral:
            norm = spectral_norm

        padding = compute_pad(5, 1)

        self.convs = nn.ModuleList([
            norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), (padding, 0))),
            norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), (padding, 0))),
            norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), (padding, 0))),
            norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), (padding, 0))),
            norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, (2,0)))
        ]) 
        self.activation = nn.LeakyReLU(0.1)

        self.post_layer = norm(nn.Conv2d(1024, 1, (3,1), 1, padding=(1,0)))

    def forward(self, x):
#         print("DiscP forward first: ", x.shape)
        feature_map = []
        batch_size, c, time = x.shape
        if time % self.p != 0:
            padding_size = self.p - (time % self.p)
            time = time + padding_size
            x = F.pad(x, (0, padding_size), "reflect")
        x = x.view(batch_size, c, time // self.p, self.p)

        for layer in self.convs:
            x = layer(x)
            x = self.activation(x)
            feature_map.append(x)
        x = self.post_layer(x)
#         feature_map.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_map
            

class MultiPeriodDiscriminator(nn.Module):

    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.d1 = DiscriminatorP(2, 5, 3, False)
        self.d2 = DiscriminatorP(3, 5, 3, False)
        self.d3 = DiscriminatorP(5, 5, 3, False)
        self.d4 = DiscriminatorP(7, 5, 3, False)
        self.d5 = DiscriminatorP(11, 5, 3, False)

    def forward(self, y_target, y_pred):
        res1_gen, fm1_gen  = self.d1(y_pred)
        res1_gt, fm1_gt = self.d1(y_target)
        
        res2_gen, fm2_gen  = self.d2(y_pred)
        res2_gt, fm2_gt = self.d2(y_target)
        
        res3_gen, fm3_gen  = self.d3(y_pred)
        res3_gt, fm3_gt = self.d3(y_target)
        
        res4_gen, fm4_gen  = self.d4(y_pred)
        res4_gt, fm4_gt = self.d4(y_target)
        
        res5_gen, fm5_gen  = self.d5(y_pred)
        res5_gt, fm5_gt = self.d5(y_target)


        gen_results = [res1_gen, res2_gen, res3_gen, res4_gen, res5_gen]
        gen_feature_maps = [fm1_gen, fm2_gen, fm3_gen, fm4_gen, fm5_gen]

        gt_results = [res1_gt, res2_gt, res3_gt, res4_gt, res5_gt]
        gt_feature_maps = [fm1_gt, fm2_gt, fm3_gt, fm4_gt, fm5_gt]

        return gt_results, gen_results, gt_feature_maps, gen_feature_maps

class DiscriminatorS(nn.Module):
    def __init__(self, normalize_spectral=False):
        super(DiscriminatorS, self).__init__()
        norm = weight_norm
        if normalize_spectral:
            norm = spectral_norm

        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.post_layer = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))
        
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        feature_map = []

        for layer in self.convs:
            x = layer(x)
            x = self.activation(x)
            feature_map.append(x)
        x = self.post_layer(x)
#         feature_map.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_map


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.d1 = DiscriminatorS(True)
        self.d2 = DiscriminatorS()
        self.d3 = DiscriminatorS()

        self.avgpooling1 = nn.AvgPool1d(4, 2, 2)
        self.avgpooling2 = nn.AvgPool1d(4, 2, 2)

    def forward(self, y_target, y_pred):
        res1_gen, fm1_gen  = self.d1(y_pred)
        res1_gt, fm1_gt = self.d1(y_target)
        y_pred, y_target = self.avgpooling1(y_pred), self.avgpooling1(y_target)
        res2_gen, fm2_gen  = self.d2(y_pred)
        res2_gt, fm2_gt = self.d2(y_target)
        y_pred, y_target = self.avgpooling2(y_pred), self.avgpooling1(y_target)
        res3_gen, fm3_gen  = self.d3(y_pred)
        res3_gt, fm3_gt = self.d3(y_target)

        gen_results = [res1_gen, res2_gen, res3_gen]
        gen_feature_maps = [fm1_gen, fm2_gen, fm3_gen]

        gt_results = [res1_gt, res2_gt, res3_gt]
        gt_feature_maps = [fm1_gt, fm2_gt, fm3_gt]

        return gt_results, gen_results, gt_feature_maps, gen_feature_maps
