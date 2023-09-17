import torch.nn as nn

class Residual(nn.module):
    def __init__(self, first):
        super().__init__()
        self.fisrt = first
    
    def forward(self, x, **kwargs):
        res = x
        out = self.fisrt(x,**kwargs)
        out += res
        return out
    
        