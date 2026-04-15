import os
import torch
import tqdm
import torchvision
from fm_net import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
pil2tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x:2 * x - 1),
])

model = UNet().to(device)
try:
    model.load_state_dict(torch.load("model.pt"))
except:
    pass
dataset = torchvision.datasets.MNIST(root="./dataset",train=True,transform=pil2tensor,download=True)
dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=128,shuffle=True)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

EPOCH = 100
model.train()
for epoch in range(EPOCH):
    Loss = 0
    for x1, label in tqdm.tqdm(dataloader):
        x1 = x1.to(device)
        label = label.to(device)

        t = torch.rand(size=(x1.size(0),)).to(device)
        x0 = torch.randn_like(x1).to(device)
        xt = (1 - t.view(-1,1,1,1)) * x0 + t.view(-1,1,1,1) * x1

        pred_vt = model(xt, t, label)

        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(pred_vt,x1 - x0)
        loss.backward()
        optimizer.step()
        Loss = loss
    torch.save(model.state_dict(),"model.pt.tmp")
    os.replace("model.pt.tmp","model.pt")
    print(f"epoch={epoch}, loss={Loss.item()}")