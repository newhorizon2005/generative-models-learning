import math
from torch import nn
from config import *


class TimePositionEmbedding(nn.Module):
    def __init__(self, time_emb_size):
        super().__init__()
        self.half_emb_size = time_emb_size // 2
        half_emb = torch.exp(
            torch.arange(self.half_emb_size) * (-math.log(10000) / (self.half_emb_size - 1))
        )
        self.register_buffer("half_emb", half_emb) # 1/10000^(i/d)

    def forward(self, t):
        t = t.view(t.size(0), 1) # b -> b, 1
        half_emb = self.half_emb.unsqueeze(0).expand(t.size(0), self.half_emb_size) # b, half_emb_size
        half_emb_t = half_emb * t # t/10000^(i/d)
        embs_t = torch.cat((half_emb_t.sin(), half_emb_t.cos()), dim=-1) # PE(t,2i)=sin(t/10000^(2i/d)), PE(t,2i+1)=cos(t/10000^(2i/d))
        return embs_t  # b, time_emb_size