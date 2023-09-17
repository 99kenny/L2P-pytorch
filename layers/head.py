import torch.nn as nn
import einops

class Head(nn.Module):
    def __init__(self,
                embed_dim : int, 
                num_class : int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_class = num_class
        
        self.norm = nn.LayerNorm(self.embed_dim) 
        self.proj =nn.Linear(self.embed_dim, self.num_class)
        
    def forward(self, x):
        x = einops.reduce(x ,'b n e -> b e', reduction="mean")
        x = self.norm(x)
        x = self.proj(x)
        return x
