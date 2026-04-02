from diffusion import *
import matplotlib.pyplot as plt
from dataset import tensor_to_pil
from dit import DiT

def backward_denoise(model,x_t,cls):
    steps = [x_t]

    global alphas,alphas_bar,variance

    model = model.to(DEVICE)
    x_t = x_t.to(DEVICE)
    alphas = alphas.to(DEVICE)
    alphas_bar = alphas_bar.to(DEVICE)
    variance = variance.to(DEVICE)
    cls = cls.to(DEVICE)

    # model.eval() 不好用，会参考历史均值
    with torch.no_grad():
        for time in range(T-1,-1,-1):
            t = torch.full((x_t.size(0),),time).to(DEVICE)
            predict_noise_t = model(x_t,t,cls)
            shape = (x_t.size(0),1,1,1)
            mean_t = 1/torch.sqrt(alphas[t].view(*shape)) * \
                     (
                         x_t - (1 - alphas[t].view(*shape))/torch.sqrt(1 - alphas_bar[t].view(*shape)) * predict_noise_t
                     )
            if time != 0:
                x_t = mean_t + torch.randn_like(x_t) * torch.sqrt(variance[t].view(*shape))
            else:
                x_t = mean_t

            x_t = torch.clamp(x_t, -1.0,1.0).detach()
            steps.append(x_t)
    return steps

def show_result():
    model = DiT(img_size=IMG_SIZE, patch_size=4, channel=CHANNEL,
                emb_size=64, label_num=CLASS_NUMBERS, dit_num=3, head=4)
    model.load_state_dict(torch.load("model.pt"))

    num_imgs = 10
    prompt = torch.tensor([28, 12, 24, 30, 29, 2, 0, 2, 5], dtype=torch.long)
    prompt = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    batch_size = len(prompt)
    x_t = torch.randn(size=(batch_size, CHANNEL, IMG_SIZE, IMG_SIZE))
    steps = backward_denoise(model, x_t, prompt)

    plt.figure(figsize=(15,15))
    for b in range(batch_size):
        for i in range(num_imgs):
            idx = int(T / num_imgs) * (i + 1)
            final_img = (steps[idx][b].to("cpu") + 1) / 2
            final_img = tensor_to_pil(final_img)
            plt.subplot(batch_size, num_imgs, b * num_imgs + i + 1)
            plt.imshow(final_img)
            plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_result()