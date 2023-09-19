import torch
import torch.nn as nn
import einops

# takes an image as an input, divide it into patches, let it through the embedding layer
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2

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

class TransformerEncoder(nn.Sequential):
    def __init__(self,
                 num_head:int,
                 depth:int,
                 embed_dim:int):
        super().__init__(*[TransformerEncoderBlock(num_head,embed_dim) for _ in range(depth)])

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
        self.cls_token = nn.Parameter(torch.randn(batch_size,1,embed_dim)).cuda()
        self.num_patch = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1,self.num_patch + 1,embed_dim)).cuda()
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