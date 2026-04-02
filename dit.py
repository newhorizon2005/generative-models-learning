import torch
import torch.nn as nn

from config import IMG_SIZE
from time_pos_emb import TimePositionEmbedding
from dit_block import DiTBlock

class DiT(nn.Module):
    def __init__(self,img_size,patch_size,channel,emb_size,
                 label_num,dit_num,head):
        super().__init__()

        self.patch_size = patch_size
        self.patch_count = img_size // patch_size
        self.channel = channel

        # patchify
        # [patch,patch,channel] -> [1,1,...]
        self.conv = nn.Conv2d(in_channels=channel,out_channels=channel*patch_size**2,
                              kernel_size=patch_size,padding=0,stride=patch_size)
        self.patch_emb = nn.Linear(in_features=channel*patch_size**2,out_features=emb_size)
        self.patch_pos_emb = nn.Parameter(torch.rand(1,self.patch_count**2,emb_size))

        # time emb
        self.time_emb = nn.Sequential(
            TimePositionEmbedding(time_emb_size=emb_size),
            nn.Linear(emb_size,emb_size),
            nn.ReLU(),
            nn.Linear(emb_size,emb_size)
        )

        # label emb
        self.label_emb = nn.Embedding(num_embeddings=label_num,embedding_dim=emb_size)

        # DiT Block
        self.dits = nn.ModuleList()
        for _ in range(dit_num):
            self.dits.append(DiTBlock(emb_size,head))

        # Layer norm
        self.ln = nn.LayerNorm(emb_size)

        # unPatchify
        self.linear = nn.Linear(emb_size,channel*patch_size**2)

    def forward(self,x,t,label):
        # x is [B,C,H,W]
        label_emb = self.label_emb(label) # [B,EMB]
        time_emb = self.time_emb(t)# [B,EMB]

        condition_emb = label_emb + time_emb

        # patch emb
        x = self.conv(x)
        x = x.permute(0,2,3,1)
        x = x.view(x.size(0),self.patch_count*self.patch_count,x.size(3))

        x = self.patch_emb(x)
        x = x + self.patch_pos_emb #[B,PATCH_COUNT**2,EMB]

        # dit blocks
        for dit in self.dits:
            x = dit(x,condition_emb)

        # Layer norm
        x = self.ln(x)

        # unPatchify
        x = self.linear(x)

        # reshape
        x = x.view(x.size(0),self.patch_count,self.patch_count,self.channel,self.patch_size,self.patch_size)
        x = x.permute(0,3,1,2,4,5) # [B,C,PC(H),PC(W),PS(H),PS(W)]
        x = x.permute(0,1,2,4,3,5) # [B,C,PC(H),PS(H),PC(W),PS(W)]
        x = x.reshape(x.size(0),self.channel,self.patch_count*self.patch_size,self.patch_count*self.patch_size)
        return x # [B,C,IMG,IMG]

if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)  # [B,C,H,W]
    t = torch.tensor([520])  # 时间步长
    label = torch.tensor([5])  # 类别标签

    dit = DiT(img_size=256, patch_size=4, channel=3, emb_size=64, label_num=10, dit_num=3, head=4)
    output = dit(x, t, label)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")