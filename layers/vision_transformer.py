import torch.nn as nn
import torch
from layers.patch_embed import PatchEmbed
from layers.transformer_encoder import TransformerEncoder
from layers.head import Head
class VisionTransformer(nn.Module):
    def __init__(self, 
                num_head : int,
                num_class : int,
                batch_size : int,
                img_size : int = 224, 
                patch_size : int = 16,
                in_channels : int = 3,
                embed_dim : int = 768,
                depth : int = 12,
                ):
        super().__init__()
        self.embedding = PatchEmbed(img_size, patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.randn(batch_size,1,embed_dim))
        self.num_patch = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1,self.num_patch + 1,embed_dim))
        self.transformer_encoder = TransformerEncoder(num_head, depth, embed_dim)
        self.head = Head(embed_dim, num_class)

    def forward(self, x): # input (img): (batch, 3, H, W)

        x = self.embedding(x) # (batch, num_patches, embed_dim)
        # add cls (batch, num_patches, embed_dim) + (batch, 1, embed_dim) = (batch,num_patches+1, embed_dim)
        x = torch.cat((self.cls_token, x), dim=1)
        # pos (batch, num_patches+1, embed_dim) + (num_patches+1, embed_dim)
        x += self.pos_embedding # (batch, num_patches+1, embed_dim)
        
        x = self.transformer_encoder(x)
        x = self.head(x)
        return x