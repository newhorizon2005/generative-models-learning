import torch
import matplotlib.pyplot as plt

from config import *
from diffusion import alphas, alphas_bar, variance
from dataset import tensor_to_pil


def backward_denoise(model, x_t, cls, cfg_scale=CFG_SCALE):
    steps = [x_t]

    _alphas = alphas.to(DEVICE)
    _alphas_bar = alphas_bar.to(DEVICE)
    _variance = variance.to(DEVICE)

    model = model.to(DEVICE)
    x_t = x_t.to(DEVICE)
    cls = cls.to(DEVICE)

    cls_uncond = torch.full_like(cls, UNCOND_LABEL)

    model.eval()
    with torch.no_grad():
        for time in range(T - 1, -1, -1):
            t = torch.full((x_t.size(0),), time, device=DEVICE)
            # 有条件预测
            noise_cond = model(x_t, t, cls)
            # 无条件预测
            noise_uncond = model(x_t, t, cls_uncond)
            # cfg
            predict_noise_t = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            # DDPM Sample
            shape = (x_t.size(0), 1, 1, 1)
            mean_t = (1 / torch.sqrt(_alphas[t].view(*shape))) * (
                x_t - (1 - _alphas[t].view(*shape)) /
                torch.sqrt(1 - _alphas_bar[t].view(*shape)) * predict_noise_t
            )

            if time != 0:
                x_t = mean_t + torch.randn_like(x_t) * torch.sqrt(_variance[t].view(*shape))
            else:
                x_t = mean_t

            x_t = torch.clamp(x_t, -1.0, 1.0).detach()
            steps.append(x_t)

    return steps


def show_result():
    num_imgs = 10
    prompt = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    batch_size = prompt.size(0)
    x_t = torch.randn(size=(batch_size, CHANNEL, IMG_SIZE, IMG_SIZE))

    model = torch.load("model.pt", weights_only=False)
    steps = backward_denoise(model, x_t, prompt, cfg_scale=CFG_SCALE)

    plt.figure(figsize=(15, 15))
    for b in range(batch_size):
        for i in range(num_imgs):
            idx = int(T / num_imgs) * (i + 1)
            final_img = (steps[idx][b].to("cpu") + 1) / 2  # [-1,1] -> [0,1]
            final_img = tensor_to_pil(final_img)
            plt.subplot(batch_size, num_imgs, b * num_imgs + i + 1)
            plt.imshow(final_img, cmap='gray')
            plt.axis('off')
    plt.suptitle(f"CFG Scale = {CFG_SCALE}", fontsize=16)
    plt.tight_layout()
    plt.show()


def compare_cfg_scales():
    digit = 3
    scales = [0, 1, 3, 7.5]
    x_t = torch.randn(size=(1, CHANNEL, IMG_SIZE, IMG_SIZE))
    model = torch.load("model.pt", weights_only=False)

    plt.figure(figsize=(len(scales) * 3, 3))
    for i, scale in enumerate(scales):
        steps = backward_denoise(model, x_t.clone(), torch.tensor([digit]), cfg_scale=scale)
        final_img = (steps[-1][0].to("cpu") + 1) / 2
        plt.subplot(1, len(scales), i + 1)
        plt.imshow(tensor_to_pil(final_img), cmap='gray')
        plt.title(f"scale={scale}")
        plt.axis('off')
    plt.suptitle(f"CFG Scale Comparison (digit={digit})", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_result()
    #compare_cfg_scales()