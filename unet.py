import torch
import torch.nn as nn

from config import *
from conv_block import ConvBlock
from time_pos_emb import TimePositionEmbedding


class UNet(nn.Module):
    def __init__(self, img_channel, channels=[64, 128, 256, 512, 1024], time_emb_size=512, q_size=16, v_size=16, f_size=32, cls_emb_size=128):
        super().__init__()
        channels = [img_channel] + channels

        self.time_emb = nn.Sequential(
            TimePositionEmbedding(time_emb_size),
            nn.Linear(time_emb_size, time_emb_size),
            nn.SiLU()
        )

        self.class_emb = nn.Embedding(CLASS_NUMBERS + 1, cls_emb_size)

        # Encoder
        # 不变尺寸，增加通道
        self.enc_conv = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.enc_conv.append(
                ConvBlock(channels[i], channels[i + 1], time_emb_size,
                          q_size, v_size, f_size, cls_emb_size)
            )

        # 池化下采样缩小一倍尺寸，最后一层除外
        self.max_pools = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.max_pools.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # Decoder
        # 放大一倍尺寸，减小一倍通道
        self.decon = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.decon.append(
                nn.ConvTranspose2d(channels[-i - 1], channels[-i - 2], kernel_size=2, stride=2)
            )

        # skip connection导致通道数翻倍，减回去
        self.dec_conv = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.dec_conv.append(
                ConvBlock(channels[-i - 1], channels[-i - 2], time_emb_size,
                          q_size, v_size, f_size, cls_emb_size)
            )

        # 还原到图片通道数
        self.output = nn.Conv2d(channels[1], img_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t, cls):
        time_emb = self.time_emb(t)
        class_emb = self.class_emb(cls)

        # Encoder path
        residual = [] # skip connections
        for i, conv in enumerate(self.enc_conv):
            x = conv(x, time_emb, class_emb)
            if i != len(self.enc_conv) - 1:
                residual.append(x)
                x = self.max_pools[i](x)

        # Decoder path
        for i, de_conv in enumerate(self.decon):
            x = de_conv(x)
            x = self.dec_conv[i](
                torch.cat((residual.pop(-1), x), dim=1),
                time_emb, class_emb
            )

        return self.output(x)