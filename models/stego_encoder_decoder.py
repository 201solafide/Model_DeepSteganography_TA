import torch.nn as nn
from models.encoder import StegoEncoder
from models.decoder import RevealNetwork

class StegoEncoderDecoder(nn.Module):
    def __init__(self, num_secrets):
        super(StegoEncoderDecoder, self).__init__()
        self.num_secrets = num_secrets
        self.encoder = StegoEncoder(num_secrets=num_secrets)
        self.decoder = RevealNetwork(num_secrets=num_secrets)

    def forward(self, cover_frame, secret_images):
        # PERBAIKAN: Menggunakan .shape[1] untuk mendapatkan jumlah secrets
        # yang benar dari tensor yang sudah di-stack.
        assert secret_images.shape[1] == self.num_secrets, "Jumlah secret images tidak sesuai dengan inisialisasi model."
        
        stego = self.encoder(cover_frame, secret_images)
        revealed = self.decoder(stego)
        return stego, revealed