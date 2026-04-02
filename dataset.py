import torch
import torchvision
from torchvision import transforms

from config import *

pil_to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()  # (H,W,C) -> (C,H,W) [0,255] -> [0,1]
])

tensor_to_pil = transforms.Compose([
    transforms.Lambda(lambda t: t * 255),
    transforms.Lambda(lambda t: t.type(torch.uint8)),
    transforms.ToPILImage()
])

dataset = torchvision.datasets.MNIST(root="dataset", train=True, download=True, transform=pil_to_tensor)