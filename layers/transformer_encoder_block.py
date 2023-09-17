import torch.nn as nn
from layers.attention import AttentionLayer
class TransformerEncoderBlock(nn.Module):
    def __init__(self, 
                 num_head : int,
                 embed_dim : int,
                 expansion : int = 4, 
                 drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attention = AttentionLayer(num_head,embed_dim)
        ## mlp
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, expansion * embed_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(expansion * embed_dim, embed_dim)
        )
    
    def forward(self, input): #
        x = self.norm1(input)
        x = self.attention(x)
        x += input
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x += residual
        return x