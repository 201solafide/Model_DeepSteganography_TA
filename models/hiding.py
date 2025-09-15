# models/hiding.py
import torch
import torch.nn as nn

class Hiding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(18, 64, kernel_size=3, padding=1)  # 3 (cover) + 15 (secret)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)  # Output stego: 3 channel

    def forward(self, cover, processed_secret):
        secret_up = nn.functional.interpolate(processed_secret, size=(448, 448), mode='bilinear')
        x = torch.cat([cover, secret_up], dim=1)  # -> [batch, 18, 448, 448]
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        return torch.sigmoid(self.conv4(x))  # Stego frame
