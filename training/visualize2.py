import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Tambahkan path ke folder utama proyek
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.stego_encoder_decoder import StegoEncoderDecoder
from utils.dataset import SteganographyDataset
from training.test import calculate_psnr

# --- Konfigurasi ---
NUM_SECRETS = 1
MODEL_PATH = f"checkpoints/model_{NUM_SECRETS}secrets_best.pth"
TEST_DATA_PATH = "data/processed/test"
NUM_EXAMPLES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Folder output visualisasi
OUTPUT_DIR = "visualisasi"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clip_img(img):
    """Memastikan nilai pixel berada pada rentang [0, 1]"""
    return np.clip(img, 0, 1)

def visualize_results():
    # 1. Dataset & DataLoader
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = SteganographyDataset(data_dir=TEST_DATA_PATH, num_secrets=NUM_SECRETS, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH}")
    
    model = StegoEncoderDecoder(num_secrets=NUM_SECRETS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    print(f"Model {MODEL_PATH} berhasil dimuat untuk visualisasi.")
    
    # 3. Ambil contoh hasil
    examples = []
    with torch.no_grad():
        for i, (covers, secrets) in enumerate(test_loader):
            if i >= NUM_EXAMPLES:
                break
            
            secrets_tensor = torch.stack(secrets, dim=1)
            covers = covers.to(DEVICE)
            secrets_tensor = secrets_tensor.to(DEVICE)
            
            stego, revealed = model(covers, secrets_tensor)
            
            example = {
                'cover': covers.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                'stego': stego.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                'secrets': [s.squeeze().permute(1, 2, 0).cpu().numpy() for s in secrets_tensor.split(1, dim=1)],
                'revealed': [r.squeeze(0).permute(1, 2, 0).cpu().numpy() for r in revealed],
                'psnr_stego_cover': calculate_psnr(stego, covers),
                'psnr_revealed_secret': [calculate_psnr(revealed[j], secrets_tensor[:, j]) for j in range(NUM_SECRETS)]
            }
            examples.append(example)

    # 4. Plot Cover vs Stego
    fig1, axes1 = plt.subplots(NUM_EXAMPLES, 2, figsize=(10, 5 * NUM_EXAMPLES))
    if NUM_EXAMPLES == 1:
        axes1 = [axes1]
    for i, example in enumerate(examples):
        ax = axes1[i]
        ax[0].imshow(clip_img(example['cover']))
        ax[0].set_title(f"Contoh {i+1} - Cover Image")
        ax[0].axis('off')
        
        ax[1].imshow(clip_img(example['stego']))
        ax[1].set_title(f"Contoh {i+1} - Stego Image\nPSNR: {example['psnr_stego_cover']:.2f} dB")
        ax[1].axis('off')
    plt.tight_layout()
    plt.suptitle('Perbandingan Kualitas: Cover vs Stego Images', fontsize=16)
    cover_stego_path = os.path.join(OUTPUT_DIR, "cover_vs_stego.png")
    plt.savefig(cover_stego_path)
    plt.show()

    # 5. Plot Secret vs Revealed
    fig2, axes2 = plt.subplots(NUM_EXAMPLES, 2 * NUM_SECRETS, figsize=(10, 5 * NUM_EXAMPLES))
    if NUM_EXAMPLES == 1:
        axes2 = [axes2]
    for i, example in enumerate(examples):
        ax_row = axes2[i]
        for j in range(NUM_SECRETS):
            ax_row[2*j].imshow(clip_img(example['secrets'][j]))
            ax_row[2*j].set_title(f"Contoh {i+1} - Secret {j+1}")
            ax_row[2*j].axis('off')
            
            ax_row[2*j+1].imshow(clip_img(example['revealed'][j]))
            ax_row[2*j+1].set_title(f"Contoh {i+1} - Revealed {j+1}\nPSNR: {example['psnr_revealed_secret'][j]:.2f} dB")
            ax_row[2*j+1].axis('off')
    plt.tight_layout()
    plt.suptitle('Perbandingan Akurasi: Secret vs Revealed Images', fontsize=16)
    secret_revealed_path = os.path.join(OUTPUT_DIR, "secret_vs_revealed.png")
    plt.savefig(secret_revealed_path)
    plt.show()

    print(f"Gambar visualisasi disimpan di folder '{OUTPUT_DIR}':")
    print(f"- {cover_stego_path}")
    print(f"- {secret_revealed_path}")

if __name__ == "__main__":
    visualize_results()
