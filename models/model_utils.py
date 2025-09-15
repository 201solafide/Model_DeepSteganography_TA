import torch
import torch.nn as nn

# Inisialisasi Bobot Jaringan
def init_weights(m):
    """
    Menginisialisasi bobot (weights) pada lapisan konvolusi
    dan menormalkan bias ke nol.
    """
    if isinstance(m, (nn.Conv2d)):
        # Menggunakan inisialisasi Kaiming untuk bobot
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        # Menetapkan bias ke nol
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d)):
        # Bobot BatchNorm diinisialisasi ke 1 dan bias ke 0
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# =====================================================================================================
# Definisi Blok Bangunan Arsitektur
# =====================================================================================================

class ConvBlock(nn.Module):
    """
    Blok Konvolusi yang terdiri dari Conv2d -> BatchNorm2d -> LeakyReLU.
    Ini adalah blok dasar untuk ekstraksi fitur.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """
    Blok Residual (Residual Block) dengan skip connection.
    Digunakan untuk melatih jaringan yang dalam dan menghindari masalah vanishing gradient.
    """
    def __init__(self, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels=out_channels, out_channels=out_channels),
            ConvBlock(in_channels=out_channels, out_channels=out_channels)
        )

    def forward(self, x):
        # Skip connection: input x ditambahkan ke output dari blok konvolusi
        return x + self.block(x)