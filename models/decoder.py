import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import ConvBlock, ResidualBlock

class RevealNetwork(nn.Module):
    def __init__(self, num_secrets): # Mengambil num_secrets sebagai argumen
        super(RevealNetwork, self).__init__()
        self.output_size = (64, 64)
        self.reveal_paths = nn.ModuleList([
            nn.Sequential(
                ConvBlock(in_channels=3, out_channels=64),
                ResidualBlock(out_channels=64),
                ResidualBlock(out_channels=64),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
            ) for _ in range(num_secrets)
        ])

    def forward(self, stego, num_secrets=None): # num_secrets dijadikan opsional
        # Jika num_secrets tidak diberikan, gunakan yang sudah ada
        if num_secrets is None:
            num_secrets = len(self.reveal_paths)
            
        secrets = []
        for i in range(num_secrets):
            out = self.reveal_paths[i](stego)
            out = F.interpolate(out, size=self.output_size, mode='bilinear')
            secrets.append(out)
        return secrets