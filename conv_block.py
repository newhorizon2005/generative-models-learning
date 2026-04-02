from torch import nn
from cross_attention import CrossAttention


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_emb_size, q_size, v_size, f_size, class_emb_size):
        super().__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1), # inc -> outc hw不变
            nn.GroupNorm(8, out_channel), # outc分8组，每组独立归一化
            nn.SiLU()
        )

        self.time_emb_linear = nn.Linear(time_emb_size, out_channel)
        self.silu = nn.SiLU()

        self.seq2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channel),
            nn.SiLU()
        )

        self.cross_attention = CrossAttention(
            channel=out_channel, q_size=q_size, v_size=v_size, f_size=f_size, class_emb_size=class_emb_size
        )

    def forward(self, x, time_emb, class_emb):
        x = self.seq1(x)
        time_emb = self.silu(self.time_emb_linear(time_emb)).view(x.size(0), x.size(1), 1, 1) # b c -> b c 1 1
        output = self.seq2(x + time_emb) # 广播 b c 1 1 + b c h w
        return self.cross_attention(output, class_emb)