# models/preparation.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Preparation(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 3 channels (RGB)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.residual = nn.Conv2d(3, 3, kernel_size=1)  # Residual path

    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return F.relu(x + residual)
