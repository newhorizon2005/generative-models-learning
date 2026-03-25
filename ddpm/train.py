import os
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import *
from dataset import dataset
from unet import UNet
from diffusion import forward_diffusion

EPOCH = 50
BATCH_SIZE = 512

dataloader = DataLoader(dataset, BATCH_SIZE, num_workers=4, persistent_workers=True, shuffle=True)

if os.path.exists("model.pt"):
    model = torch.load("model.pt", weights_only=False)
else:
    model = UNet(CHANNEL).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
Loss = nn.MSELoss()

if __name__ == "__main__":
    model.train()
    for epoch in range(EPOCH + 1):
        total_loss = 0
        count = 0
        for x, cls in tqdm.tqdm(dataloader, desc=f"Epoch {epoch}"):
            x = x.to(DEVICE) * 2 - 1  # [0,1] -> [-1,1]
            t = torch.randint(0, T, (x.size(0),)).to(DEVICE)
            cls = cls.to(DEVICE)

            drop_mask = torch.rand(x.size(0), device=DEVICE) < UNCOND_PROB
            cls = torch.where(drop_mask, torch.full_like(cls, UNCOND_LABEL), cls)

            x_t, noise_t = forward_diffusion(x, t)
            predict_t = model(x_t, t, cls)
            loss = Loss(predict_t, noise_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            count += x.size(0)

        avg_loss = total_loss / count
        print(f"epoch:{epoch} avg_loss:{avg_loss:.6f}")
        torch.save(model, "model.pt.tmp")
        os.replace("model.pt.tmp", "model.pt")