import torch
import torch.nn as nn

# takes an image as an input, divide it into patches, let it through the embedding layer
class PatchEmbed(nn.Module):
    def __init__(self,
                img_size : int,
                patch_size : int,
                in_channels : int =3,
                embed_dim : int =768):
        super().__init__()
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            stride=patch_size, 
            kernel_size=patch_size
        )
        
    def forward(self, x):
        x = self.proj(x)               # (n, 3, 224, 224) -> (n, embed_dim, patch_size, patch_size, ) 
        x = torch.flatten(x, start_dim=2, end_dim=3)  # (n, embed_dim, patch_size*patch_size)
        x = torch.transpose(x,1,2)           # (n, patch_size*patch_size, embed_dim)
        return x