import torch
import torch.nn as nn
import torch.nn.functional as F  # Tambahkan impor ini
from models.model_utils import ConvBlock, ResidualBlock

class PreparationNetwork(nn.Module):
    def __init__(self, num_secrets): # Mengambil num_secrets sebagai argumen
        super(PreparationNetwork, self).__init__()
        self.secret_paths = nn.ModuleList([
            nn.Sequential(
                ConvBlock(in_channels=3, out_channels=64),
                ResidualBlock(out_channels=64),
                ResidualBlock(out_channels=64)
            ) for _ in range(num_secrets)
        ])

    def forward(self, secrets):
        # Tambahkan assert untuk memastikan jumlah secrets sesuai
        assert len(secrets) == len(self.secret_paths), "Jumlah secret images tidak sesuai dengan arsitektur PreparationNetwork."
        
        outputs = []
        for i in range(len(secrets)):
            feat = self.secret_paths[i](secrets[i])
            outputs.append(feat)
        merged = torch.cat(outputs, dim=1)
        return merged

class HidingNetwork(nn.Module):
    def __init__(self, num_secrets, out_channels=3):
        super(HidingNetwork, self).__init__()
        in_channels = 3 + 64 * num_secrets
        self.network = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=64),
            ResidualBlock(out_channels=64),
            ResidualBlock(out_channels=64),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        )
# F.interpolate(...): Fungsi ini mengubah ukuran secret_feat menjadi cover_size yang diinginkan

    def forward(self, cover, secret_feat):
        # PERBAIKAN: Resize secret_feat agar sesuai dengan ukuran cover
        # Ambil ukuran spasial dari cover (height, width)
        cover_size = cover.shape[2:]
        # Lakukan interpolasi pada secret_feat agar ukurannya sama dengan cover
        secret_feat_resized = F.interpolate(secret_feat, size=cover_size, mode='bilinear', align_corners=False)

        x = torch.cat([cover, secret_feat_resized], dim=1)
        return self.network(x)

class StegoEncoder(nn.Module):
    def __init__(self, num_secrets):
        super(StegoEncoder, self).__init__()
        self.prep_network = PreparationNetwork(num_secrets=num_secrets)
        self.hide_network = HidingNetwork(num_secrets=num_secrets)

    def forward(self, cover, secrets):
        # PERBAIKAN: Mengubah tensor secrets [B, N, C, H, W]
        # menjadi list of tensors [B, C, H, W] * N
        # secrets adalah tensor (batch, num_secrets, channels, height, width)
        # kita perlu mengubahnya menjadi list of tensors, di mana setiap
        # elemen list adalah satu secret tensor (batch, channels, height, width)
        secrets_list = [secrets[:, i, :, :, :] for i in range(secrets.shape[1])]
        
        secret_feat = self.prep_network(secrets_list)
        stego = self.hide_network(cover, secret_feat)
        return stego