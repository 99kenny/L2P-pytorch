import torch
import torch.nn as nn
import einops
import numpy as np
def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class AttentionLayer(nn.Module):
    def __init__(self,
                 num_head : int,
                 dim : int,
                 dropout : float,
                 qkv_bias : bool = True):
        super().__init__()
        self.num_head = num_head
        self.dim = dim
        #self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj_q = nn.Linear(dim,dim)
        self.proj_k = nn.Linear(dim,dim)
        self.proj_v = nn.Linear(dim,dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # input : (n, num_patch+1, embed_dim)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.num_head, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = self.drop(nn.Softmax(dim=-1)(scores))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores

        return h

class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim, ff):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff)
        self.fc2 = nn.Linear(ff, dim)

    def forward(self, x):
        return self.fc2(nn.GELU()(self.fc1(x)))
    
class Block(nn.Module):
    def __init__(self,
                 num_head : int,
                 embed_dim : int,
                 expansion : int = 4,
                 drop: float = 0.):
        super().__init__()
        self.attn = AttentionLayer(num_head, embed_dim, drop)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
        ## mlp
        self.pwff = PositionWiseFeedForward(embed_dim, embed_dim * expansion)
        self.drop = nn.Dropout(drop)

    def forward(self, x): #
        x = x + self.drop(self.proj(self.attn(self.norm1(x))))
        x = x + self.drop(self.pwff(self.norm2(x)))

        return x

class TransformerEncoder(nn.Module):
    def __init__(self,
                 num_head:int,
                 depth:int,
                 embed_dim:int,
                 dropout:float):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(num_head, embed_dim, drop=dropout) for _ in range(depth)
        ])
        
    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x)
        return x
        
class VisionTransformer(nn.Module):
    def __init__(self,
                num_head : int,
                num_class : int,
                img_size : int = 224,
                patch_size : int = 16,
                in_channels : int = 3,
                embed_dim : int = 768,
                depth : int = 12,
                dropout_rate : float = 0.
                ):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels,
            embed_dim,
            stride=patch_size,
            kernel_size=patch_size
        )
        self.class_token = nn.Parameter(torch.randn(1,1,embed_dim))
        num_patch = (img_size // patch_size) ** 2
        self.positional_embedding = nn.Parameter(torch.randn(1,num_patch + 1,embed_dim))
        
        
        self.transformer = TransformerEncoder(num_head, depth, embed_dim, dropout_rate)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.fc = nn.Linear(embed_dim, num_class)
        
        #self.init_weights()
    def forward(self, x): # input (img): (batch, 3, H, W)
        
        b, c, h, w = x.shape
        x = self.patch_embedding(x) # (batch, num_patches, embed_dim)
        x = torch.flatten(x, start_dim=2, end_dim=3)  # (n, embed_dim, patch_size*patch_size)
        x = torch.transpose(x,1,2)           # (n, patch_size*patch_size, embed_dim)
    
        # add cls (batch, num_patches, embed_dim) + (batch, 1, embed_dim) = (batch,num_patches+1, embed_dim)
        x = torch.cat((self.class_token.expand(b,-1,-1), x), dim=1) 
            
        # pos (batch, num_patches+1, embed_dim) + (num_patches+1, embed_dim)
        x = x + self.positional_embedding
        
        x = self.transformer(x)
        
        x = self.norm(x)[:,0] #cls
        x = self.fc(x)
        
        return x