import torch
import matplotlib.pyplot as plt
from fm_net import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)
model.load_state_dict(torch.load("model.pt"))
x = torch.randn(size=(1,1,28,28)).to(device)
steps = 250
label = 9

model.eval()
with torch.no_grad():
    for i in range(steps):
        t = torch.tensor([1.0/steps*i]).to(device)
        label = torch.tensor([label],dtype=torch.long).to(device)
        pred_vt = model(x,t,label)
        x = x+pred_vt*1.0/steps
        x = x.detach()

x = (x+1)/2
plt.figure(figsize=(1,1))
plt.axis("off")
plt.imshow(x[0,0].cpu().numpy(),cmap="gray")
plt.show()