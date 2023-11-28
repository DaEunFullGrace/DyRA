from detectron2.utils.logger import setup_logger
import argparse, math, warnings, logging, os
import numpy as np
from itertools import chain, compress
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SimplePredictor", 
           "bilinear_interpolation"]


def _fp_linspace(step, st=-1, end=1):
    step = ((end-st)/(step-1)).expand(int((step).tolist()-1))
    step = torch.cat((torch.tensor([0.]).to(step.device), step), 0)
    step = torch.cumsum(step, dim=0)
    linsp = step - torch.ones(step.shape).to(step.device)
    return linsp
    
def bilinear_interpolation(img, sizes):
    assert img.dim() >= 3
    if img.dim() == 3:
        img = img.unsqueeze(0)
    B = img.shape[0]
    img.requires_grad = False
    
    lin_x, lin_y = _fp_linspace(sizes[-2]), _fp_linspace(sizes[-1])
    grid_w, grid_h = torch.meshgrid(lin_x, lin_y)
    
    grid = torch.stack([grid_h, grid_w], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1).to(img.device)
    output = F.grid_sample(img.float(), grid.float(), align_corners=True)    
    return output

class Multiply(nn.Module):
    def __init__(self, gain=2):
        super(Multiply, self).__init__()
        self.gain = gain
        
    def forward(self, x):
        # return torch.clamp(self.gain*x, max=self.gain) # min=min_val, 
        return torch.mul(x, self.gain).clamp(min=self.gain**2 * 0.1)


################### network ###################
class SimplePredictor(nn.Module):
    def __init__(self, in_chan, out_chan=1, gain=2, grid_len=3, num_mlayer=1024, device='cuda'):
        super().__init__()
        self.device = device
        self.gain = gain
        self.num_mlayer = num_mlayer
        self.cvt_layers = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Linear(num_mlayer, in_chan),)
        
        self.enc_layers = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=in_chan, nhead=int(in_chan/grid_len)),
                num_layers=3),
        )
        
        self.out_layers = nn.Sequential(nn.Sigmoid(), Multiply(gain))
        self.fc_layer = nn.Sequential(nn.Linear(in_chan, 1),)   # w,h
        
        self._init_weights(self.cvt_layers)
        self._init_weights(self.enc_layers)
        self._init_weights(self.fc_layer)
    
    def _init_weights(self, net, init_val=0.0):
        for m in net:
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # m.weight.data.normal_(mean=init_val, std=0.01)
                nn.init.xavier_normal(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Transformer):
                # m.weight.data.normal_(mean=init_val, std=0.01)
                nn.init.xavier_normal(m.weight)
                m.bias.data.zero_()
        net.train()
        net.to(self.device)
        
    def forward(self, x):
        x = self.cvt_layers(x)
        x = self.enc_layers(x)
        res_sf = self.out_layers(self.fc_layer(x))
        return res_sf