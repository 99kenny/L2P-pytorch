import torch
import torch.nn as nn
import einops

class AttentionLayer(nn.Module):
    def __init__(self, 
                 num_head : int,
                 dim : int, 
                 qkv_bias : bool = True):
        super().__init__()
        self.num_head = num_head
        self.dim = dim
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.dropout_attn = nn.Dropout(0.)
        self.proj = nn.Linear(dim, dim)
        self.dropout_proj = nn.Dropout(0.)
        
    def forward(self, x):
        # input : (n, num_patch+1, embed_dim)
        
        num_samples, num_tokens, dim = x.shape
        if dim != self.dim :
            raise ValueError
        
        # multi-head attention
        # create qkv
        x = self.qkv(x)
        # decouple q k v 
        x = einops.rearrange(x, 'b  n (h d qkv) -> (qkv) b h n d', h=self.num_head, qkv=3)
        q,k,v = x[0], x[1], x[2]
        
        # Scaled dot product attention
        # q * k / d ** 1/2
        e = torch.einsum('bnqd, bnkd -> bnqk', q, k) / ((self.dim)**(1/2))
        x = nn.Softmax()(e)
        # * v
        x = torch.einsum('bnqk, bnvd -> bnqd',x,v)
        x = einops.rearrange(x,'b n q d -> b q (n d)')
        # linear 
        x = self.proj(x)
        
        return x