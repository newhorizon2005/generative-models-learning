from matplotlib import pyplot as plt
from config import *
from dataset import dataset, tensor_to_pil

betas = torch.linspace(0.0001,0.02,T)
alphas = 1 - betas

alphas_bar = torch.cumprod(alphas,dim=-1)
# 去掉最后一个，在前面补1.0 -> 类似于右移1个，左侧补1.0
alphas_bar_prev = torch.cat((torch.tensor([1.0]),alphas_bar[:-1]),dim=-1)
# variance方差
variance = (1 - alphas) * (1 - alphas_bar_prev) / (1 - alphas_bar)

# x is like [batch,channel,width,height]
# t is like [batch]
def forward_diffusion(x,t):

    # noise_t is like [batch,channel,width,height]
    noise_t = torch.randn_like(x)

    # alphas_t_bar is like [batch,1,1,1]
    # 形状要对齐，乘法是按像素相乘
    alphas_t_bar = alphas_bar.to(DEVICE)[t].view(x.size(0),1,1,1)
    x_t = torch.sqrt(alphas_t_bar) * x + torch.sqrt(1 - alphas_t_bar) * noise_t

    return x_t, noise_t

if __name__ == "__main__":
    # [2,1,48,48]
    # [B,C,W,H]
    x = torch.stack((dataset[0][0], dataset[1][0]), dim=0).to(DEVICE)

    # before
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(tensor_to_pil(x[0]))
    plt.subplot(1,2,2)
    plt.imshow(tensor_to_pil(x[1]))
    plt.show()

    x = x * 2 - 1 # [0,1] -> [-1,1] 拟合高斯数值分布
    t = torch.randint(0,T,size=(x.size(0),)).to(DEVICE)
    x_t, noise_t = forward_diffusion(x,t)

    # after
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(tensor_to_pil((x_t[0]+1)/2))
    plt.subplot(1,2,2)
    plt.imshow(tensor_to_pil((x_t[1]+1)/2))
    plt.show()