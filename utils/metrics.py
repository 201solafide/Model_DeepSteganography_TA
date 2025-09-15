import torch

def calculate_psnr(img1, img2, max_pixel=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_pixel ** 2 / mse).item()