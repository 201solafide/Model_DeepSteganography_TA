import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import matplotlib.pyplot as plt
from tqdm import tqdm # Import tqdm

from models.stego_encoder_decoder import StegoEncoderDecoder
from utils.dataset import SteganographyDataset
from models.model_utils import init_weights
from utils.checkpoint_utils import save_checkpoint, load_checkpoint

# --- Konfigurasi Pelatihan ---
NUM_SECRETS = 1       # Ganti 1, 2, 3, 4, atau 5
EPOCHS = 200
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = f"checkpoints/model_{NUM_SECRETS}secrets_checkpoint.pth.tar"
FINAL_MODEL_PATH = f"checkpoints/model_{NUM_SECRETS}secrets_final.pth"
PLOT_PATH = f"plots/loss_plot_{NUM_SECRETS}secrets.png"

# Parameter Loss Function
ALPHA = 1.0
BETA = 1.0

def train(num_secrets):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = SteganographyDataset(data_dir="data/processed/train", num_secrets=num_secrets, transform=data_transforms)
    val_dataset = SteganographyDataset(data_dir="data/processed/val", num_secrets=num_secrets, transform=data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # num_workers=4, contoh
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = StegoEncoderDecoder(num_secrets=num_secrets).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Tambahkan variabel untuk melacak loss validasi terbaik
    best_val_loss = float('inf')
    BEST_MODEL_PATH = f"checkpoints/model_{num_secrets}secrets_best.pth"

    start_epoch = 0
    # Jika ada checkpoint, muat dan lanjutkan
    if os.path.exists(CHECKPOINT_PATH):
        model, optimizer, start_epoch = load_checkpoint(CHECKPOINT_PATH, model, optimizer)
        # TODO: Muat best_val_loss dari checkpoint jika ada
    else:
        model.apply(init_weights)

    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, EPOCHS):
        # --- Pelatihan ---
        model.train()
        running_loss = 0.0
        
        # Menggunakan tqdm untuk loop pelatihan
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Training", unit="batch")
        for covers, secrets in pbar:
            secrets_tensor = torch.stack(secrets, dim=1)
            
            covers = covers.to(DEVICE)
            secrets_tensor = secrets_tensor.to(DEVICE)
            
            optimizer.zero_grad()
            
            stego, revealed = model(covers, secrets_tensor)
            
            loss_cover = criterion(stego, covers)
            loss_secret = 0.0
            
            for j in range(num_secrets):
                loss_secret += criterion(revealed[j], secrets_tensor[:, j])
            
            loss_secret /= num_secrets
            
            total_loss = ALPHA * loss_cover + BETA * loss_secret
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            # Update deskripsi tqdm dengan loss saat ini
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        avg_train_loss = running_loss / len(train_loader)
        
        # --- Validasi ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            # Menggunakan tqdm untuk loop validasi
            pbar_val = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Validation", unit="batch")
            for covers, secrets in pbar_val:
                secrets_tensor = torch.stack(secrets, dim=1)
                
                covers = covers.to(DEVICE)
                secrets_tensor = secrets_tensor.to(DEVICE)
                
                stego, revealed = model(covers, secrets_tensor)
                
                loss_cover = criterion(stego, covers)
                loss_secret = 0.0
                
                for j in range(num_secrets):
                    loss_secret += criterion(revealed[j], secrets_tensor[:, j])
                
                loss_secret /= num_secrets
                
                total_loss = ALPHA * loss_cover + BETA * loss_secret
                running_val_loss += total_loss.item()
                
                pbar_val.set_postfix({'val_loss': f'{total_loss.item():.4f}'})

        avg_val_loss = running_val_loss / len(val_loader)

        # Simpan nilai loss untuk plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- Cek dan Simpan Model Terbaik ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Model terbaik disimpan di: {BEST_MODEL_PATH} dengan Val Loss: {best_val_loss:.4f}")

        # --- Simpan Checkpoint Setiap Akhir Epoch (untuk resume) ---
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, filename=CHECKPOINT_PATH)
    
    print("Pelatihan selesai!")
    os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Model final disimpan di: {FINAL_MODEL_PATH}")

    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss for {num_secrets} Secrets')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_PATH)
    print(f"Grafik loss disimpan di: {PLOT_PATH}")

if __name__ == "__main__":
    train(num_secrets=NUM_SECRETS)