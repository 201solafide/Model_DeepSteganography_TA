import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import os
import sys

# Tambahkan path ke folder utama proyek
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.stego_encoder_decoder import StegoEncoderDecoder
from utils.dataset import SteganographyDataset

# --- Konfigurasi Pengujian ---
# Ganti dengan jumlah secret yang Anda uji (misal: 1, 2, 3, 4, 5)
NUM_SECRETS = 1 
# Gunakan model terbaik yang telah disimpan selama pelatihan
MODEL_PATH = f"checkpoints/model_1secrets_checkpoint.pth.tar" 
TEST_DATA_PATH = "data/processed/test"
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_psnr(img1, img2):
    """Menghitung PSNR antara dua gambar. img1 dan img2 harus tensor dengan nilai piksel [0, 1]."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10: # Hindari logaritma dari nol, tambahkan toleransi kecil
        return 100
    max_pixel = 1.0 # Karena gambar dinormalisasi ke [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def calculate_bpp(num_secrets, cover_size=(448, 448), secret_size=(64, 64), num_channels=3):
    """Menghitung kapasitas bit per pixel (BPP)."""
    cover_pixels = cover_size[0] * cover_size[1]
    secret_pixels = secret_size[0] * secret_size[1]
    total_secret_bits = num_secrets * secret_pixels * num_channels * 8
    bpp = total_secret_bits / cover_pixels
    return bpp

def test_model():
    # 1. Siapkan Dataset dan DataLoader
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_dataset = SteganographyDataset(data_dir=TEST_DATA_PATH, num_secrets=NUM_SECRETS, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Muat Model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH}")
    
    model = StegoEncoderDecoder(num_secrets=NUM_SECRETS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print(f"Model {MODEL_PATH} berhasil dimuat untuk pengujian.")
    
    # 3. Pengujian dan Penghitungan Metrik
    psnr_stego_cover_list = []
    psnr_revealed_secret_list = []
    inference_times = []
    
    with torch.no_grad():
        for covers, secrets in test_loader:
            secrets_tensor = torch.stack(secrets, dim=1)
            
            covers = covers.to(DEVICE)
            secrets_tensor = secrets_tensor.to(DEVICE)
            
            # Pengukuran Waktu Komputasi
            start_time = time.time()
            stego, revealed = model(covers, secrets_tensor)
            torch.cuda.synchronize() # Tunggu operasi GPU selesai
            end_time = time.time()
            
            # Hitung waktu inferensi per sampel
            inference_time_ms = (end_time - start_time) * 1000
            inference_times.append(inference_time_ms)
            
            # Hitung PSNR (Stego vs Cover)
            psnr_stego_cover = calculate_psnr(stego, covers)
            psnr_stego_cover_list.append(psnr_stego_cover)
            
            # Hitung PSNR (Revealed vs Secret)
            avg_psnr_revealed_secret = 0.0
            for j in range(NUM_SECRETS):
                psnr_revealed_secret = calculate_psnr(revealed[j], secrets_tensor[:, j])
                avg_psnr_revealed_secret += psnr_revealed_secret
            
            avg_psnr_revealed_secret /= NUM_SECRETS
            psnr_revealed_secret_list.append(avg_psnr_revealed_secret)
            
    # 4. Tampilkan Hasil Akhir
    avg_psnr_stego_cover = np.mean(psnr_stego_cover_list)
    avg_psnr_revealed_secret = np.mean(psnr_revealed_secret_list)
    avg_inference_time = np.mean(inference_times)
    
    bpp = calculate_bpp(NUM_SECRETS)
    
    print("--- Hasil Pengujian Akhir ---")
    print(f"Konfigurasi Model: {NUM_SECRETS} Secret Images")
    print(f"Kapasitas Penyisipan (BPP): {bpp:.2f} bits per pixel")
    print(f"Rata-rata PSNR (Stego vs Cover): {avg_psnr_stego_cover:.2f} dB")
    print(f"Rata-rata PSNR (Revealed vs Secret): {avg_psnr_revealed_secret:.2f} dB")
    print(f"Waktu Komputasi Rata-rata per Sampel: {avg_inference_time:.4f} ms")

if __name__ == "__main__":
    test_model()