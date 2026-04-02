import torch

T = 1000
CHANNEL = 1
IMG_SIZE = 32
CLASS_NUMBERS = 10
CFG_SCALE = 7.5
UNCOND_PROB = 0.1
UNCOND_LABEL = CLASS_NUMBERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"