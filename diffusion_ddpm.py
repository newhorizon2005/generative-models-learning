from config import *
from matplotlib import pyplot as plt
from dataset import dataset, tensor_to_pil

betas = torch.linspace(0.0001, 0.02, T)
alphas = 1 - betas

alphas_bar = torch.cumprod(alphas, dim=-1)
alphas_bar_prev = torch.cat((torch.tensor([1.0]), alphas_bar[:-1]), dim=-1) # 右移1位，左侧补1.0
variance = (1 - alphas) * (1 - alphas_bar_prev) / (1 - alphas_bar) # 后验方差 q(x_{t-1}|x_t, x_0)

def forward_diffusion(x, t):
    noise = torch.randn_like(x)
    alphas_t_bar = alphas_bar.to(DEVICE)[t].view(x.size(0), 1, 1, 1)
    x_t = torch.sqrt(alphas_t_bar) * x + torch.sqrt(1 - alphas_t_bar) * noise
    return x_t, noise


if __name__ == "__main__":
    x = torch.stack((dataset[0][0], dataset[1][0]), dim=0).to(DEVICE)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_to_pil(x[0]), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(tensor_to_pil(x[1]), cmap='gray')
    plt.suptitle("Before diffusion")
    plt.show()

    x = x * 2 - 1  # [0,1] -> [-1,1]
    t = torch.randint(0, T, size=(x.size(0),)).to(DEVICE)
    x_t, noise_t = forward_diffusion(x, t)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_to_pil((x_t[0] + 1) / 2), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(tensor_to_pil((x_t[1] + 1) / 2), cmap='gray')
    plt.suptitle(f"After diffusion t={t.tolist()}")
    plt.show()