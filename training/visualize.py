import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Tambahkan path ke folder utama proyek
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.stego_encoder_decoder import StegoEncoderDecoder
from utils.dataset import SteganographyDataset
from training.test import calculate_psnr

# --- Konfigurasi Visualisasi ---
NUM_SECRETS = 1 
MODEL_PATH = f"checkpoints/model_{NUM_SECRETS}secrets_best.pth" 
TEST_DATA_PATH = "data/processed/test"
NUM_EXAMPLES = 3 # Jumlah contoh yang akan divisualisasikan
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def visualize_results():
    # 1. Siapkan Dataset dan DataLoader
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Gunakan DataLoader dengan shuffle=False agar hasilnya konsisten
    test_dataset = SteganographyDataset(data_dir=TEST_DATA_PATH, num_secrets=NUM_SECRETS, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. Muat Model Terbaik
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH}")
    
    model = StegoEncoderDecoder(num_secrets=NUM_SECRETS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print(f"Model {MODEL_PATH} berhasil dimuat untuk visualisasi.")
    
    # 3. Visualisasi Hasil
    fig, axes = plt.subplots(NUM_EXAMPLES, 3 + NUM_SECRETS, figsize=(15, 5 * NUM_EXAMPLES))
    if NUM_EXAMPLES == 1:
        axes = [axes] # Pastikan axes selalu berupa array
    
    with torch.no_grad():
        for i, (covers, secrets) in enumerate(test_loader):
            if i >= NUM_EXAMPLES:
                break
            
            secrets_tensor = torch.stack(secrets, dim=1)
            
            covers = covers.to(DEVICE)
            secrets_tensor = secrets_tensor.to(DEVICE)
            
            stego, revealed = model(covers, secrets_tensor)
            
            # Pindahkan tensor ke CPU dan ubah ke format gambar
            cover_img = covers.squeeze(0).permute(1, 2, 0).cpu().numpy()
            stego_img = stego.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Hitung PSNR untuk display
            psnr_stego = calculate_psnr(stego, covers)
            psnr_revealed_list = []
            for j in range(NUM_SECRETS):
                psnr_revealed_list.append(calculate_psnr(revealed[j], secrets_tensor[:, j]))

            # Tampilkan Gambar
            ax = axes[i]
            
            # Cover Image
            ax[0].imshow(cover_img)
            ax[0].set_title(f"Cover Image")
            ax[0].axis('off')
            
            # Stego Image
            ax[1].imshow(stego_img)
            ax[1].set_title(f"Stego Image\nPSNR: {psnr_stego:.2f} dB")
            ax[1].axis('off')
            
            # Decoded Secrets
            for j in range(NUM_SECRETS):
                secret_img = secrets_tensor[:, j].squeeze(0).permute(1, 2, 0).cpu().numpy()
                revealed_img = revealed[j].squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                # Tampilkan Secret Asli
                ax[2 + j].imshow(secret_img)
                ax[2 + j].set_title(f"Secret {j+1}")
                ax[2 + j].axis('off')
                
                # Tampilkan Revealed Secret
                if len(ax) > 3: # Cek jika ada kolom tambahan untuk revealed
                    ax[3 + j].imshow(revealed_img)
                    ax[3 + j].set_title(f"Revealed {j+1}\nPSNR: {psnr_revealed_list[j]:.2f} dB")
                    ax[3 + j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_results()