import torch
import torch.nn as nn
import einops

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
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_proj = nn.Dropout(dropout)

    def forward(self, x):
        # input : (n, num_patch+1, embed_dim)
        num_samples, num_tokens, dim = x.shape
        if dim != self.dim :
            raise ValueError

        
        q,k,v = self.proj_q(x), self.proj_k(x), self.proj_q(x)
        # Scaled dot product attention
        # q * k / d ** 1/2
        e = torch.einsum('bnqd, bnkd -> bnqk', q, k) / ((self.dim)**(1/2))
        x = nn.Softmax()(e)
        # * v
        x = torch.einsum('bnqk, bnvd -> bnqd',x,v)
        x = einops.rearrange(x,'b n q d -> b q (n d)')

        return x

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
        self.pwff = nn.Sequential(
            nn.Linear(embed_dim, expansion * embed_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(expansion * embed_dim, embed_dim)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x): #
        x = x + self.proj(self.attention(self.norm1(x)))
        x = x + self.pwff(self.norm2(x))

        return x

class Head(nn.Module):
    def __init__(self,
                embed_dim : int,
                num_class : int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_class = num_class
        self.norm = nn.LayerNorm(self.embed_dim)
        self.proj = nn.Linear(self.embed_dim, self.num_class)
    def forward(self, input):
        x = einops.reduce(input ,'b n e -> b e', reduction="mean")
        x = self.norm(x)
        x = self.proj(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self,
                 num_head:int,
                 depth:int,
                 embed_dim:int,
                 dropout:float):
        self.blocks = nn.ModuleList([
            Block(num_head, embed_dim, drop=dropout) for _ in range(depth)
        ])
        
    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
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
        self.positional_embedding = nn.Parameter(torch.randn(1,self.num_patch + 1,embed_dim))
        self.num_patch = (img_size // patch_size) ** 2
        
        
        self.transformer = TransformerEncoder(num_head, depth, embed_dim, dropout_rate)
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.fc = nn.Linear(self.embed_dim, num_class)
        
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