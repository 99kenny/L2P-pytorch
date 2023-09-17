import torch.nn as nn
from layers.transformer_encoder_block import TransformerEncoderBlock

class TransformerEncoder(nn.Sequential):
    def __init__(self,
                 num_head:int,
                 depth:int,
                 embed_dim:int):
        super().__init__(*[TransformerEncoderBlock(num_head,embed_dim) for _ in range(depth)])
    