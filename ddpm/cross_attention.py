import math
import torch
import torch.nn as nn
from config import *


class CrossAttention(nn.Module):
    def __init__(self, channel, q_size, v_size, f_size, class_emb_size, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens

        self.class_to_tokens = nn.Linear(class_emb_size, num_tokens * q_size)

        self.w_q = nn.Linear(channel, q_size)
        self.w_k = nn.Linear(q_size, q_size)
        self.w_v = nn.Linear(q_size, v_size)

        self.scale = math.sqrt(q_size)

        self.z_linear = nn.Linear(v_size, channel)
        self.norm1 = nn.LayerNorm(channel)
        self.feedforward = nn.Sequential(
            nn.Linear(channel, f_size),
            nn.ReLU(),
            nn.Linear(f_size, channel)
        )
        self.norm2 = nn.LayerNorm(channel)

    def forward(self, x, class_emb):
        B, C, H, W = x.shape # [B, C, H, W]
        x_p = x.permute(0, 2, 3, 1)  # [B, H, W, C]
 
        Q = self.w_q(x_p).view(B, H * W, -1) # [B, H*W, q_size]

        # [B, class_emb_size] -> [B, num_tokens * q_size] -> [B, num_tokens, q_size]
        cls_tokens = self.class_to_tokens(class_emb).view(B, self.num_tokens, -1)

        K = self.w_k(cls_tokens) # [B, num_tokens, q_size]
        V = self.w_v(cls_tokens) # [B, num_tokens, v_size]

        # attention: [B, H*W, num_tokens]
        attn = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        attn = torch.softmax(attn, dim=-1)

        # Z: [B, H*W, v_size]
        Z = torch.matmul(attn, V)
        Z = self.z_linear(Z)  # [B, H*W, C]
        Z = Z.view(B, H, W, C)

        # residual + LayerNorm + FFN
        Z = self.norm1(Z + x_p)
        out = self.feedforward(Z)
        out = self.norm2(out + Z)
        return out.permute(0, 3, 1, 2)  # [B, C, H, W]