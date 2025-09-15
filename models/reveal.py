# models/reveal.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Reveal(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2)   # 448 → 224
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2) # 224 → 112
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)# 112 → 56
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1) # 56 → 56
        self.upsample = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
        self.output = nn.Conv2d(64, 15, kernel_size=3, padding=1)           # Output: (B, 15, 64, 64)

    def forward(self, stego):
        x = F.relu(self.conv1(stego))  # (B, 64, 224, 224)
        x = F.relu(self.conv2(x))      # (B, 128, 112, 112)
        x = F.relu(self.conv3(x))      # (B, 128, 56, 56)
        x = F.relu(self.conv4(x))      # (B, 64, 56, 56)
        x = self.upsample(x)           # (B, 64, 64, 64)
        x = torch.sigmoid(self.output(x))  # (B, 15, 64, 64)
        return torch.chunk(x, 5, dim=1)     # list of 5 × (B, 3, 64, 64)
