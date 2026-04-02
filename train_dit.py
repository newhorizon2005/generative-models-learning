import tqdm
from config import *
from torch.utils.data import DataLoader
from dataset import train_dataset
from dit import DiT
from diffusion import forward_diffusion
import torch.nn as nn
import os
from denoise import show_result

EPOCH = 200
BATCH_SIZE = 400

dataloader = DataLoader(train_dataset,BATCH_SIZE,
                        num_workers=4,persistent_workers=True,shuffle=True)

model = DiT(img_size=IMG_SIZE,patch_size=4,channel=CHANNEL,
            emb_size=64,label_num=CLASS_NUMBERS,dit_num=3,head=4).to(DEVICE)

try:
    model.load_state_dict(torch.load("model.pt"))
except:
    pass

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
Loss = nn.L1Loss()

if __name__ == "__main__":
    print(DEVICE)
    print(f"parameters:{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.train()
    for epoch in range(EPOCH + 1):
        bar = tqdm.tqdm(dataloader)
        for x, cls in bar:
            x = x.to(DEVICE) * 2 - 1
            t = torch.randint(0,T,(x.size(0),)).to(DEVICE)
            cls = cls.to(DEVICE)
            x_t, noise_t = forward_diffusion(x,t)
            predict_t = model(x_t,t,cls)
            loss = Loss(predict_t, noise_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.set_postfix({'epoch':f'{epoch}','loss': f'{loss.item():.6f}'})
        torch.save(model.state_dict(), "model.pt.tmp")
        os.replace("model.pt.tmp", "model.pt")
        if epoch % 50 == 0 : show_result()