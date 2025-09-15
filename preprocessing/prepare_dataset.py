import os
import shutil
import random
from PIL import Image
import cv2

# --- Konfigurasi ---
RAW_DATA_PATH = "data/raw"
COVER_IMAGE_FOLDER = "covers"
SECRET_IMAGE_FOLDER = "secrets"
PROCESSED_DATA_PATH = "data/processed"

TRAIN_SPLIT = 0.8  # 80% untuk pelatihan
VAL_SPLIT = 0.1    # 10% untuk validasi
TEST_SPLIT = 0.1   # 10% untuk pengujian

COVER_TARGET_SIZE = (448, 448)  # Ukuran target cover frame
SECRET_TARGET_SIZE = (64, 64)    # Ukuran target secret image

def resize_and_save(file_path, save_path, target_size):
    """Fungsi untuk resize gambar dan menyimpannya."""
    try:
        img = Image.open(file_path).convert("RGB")
        img_resized = img.resize(target_size, Image.LANCZOS)
        img_resized.save(save_path)
    except Exception as e:
        print(f"Gagal memproses file {file_path}: {e}")

def prepare_dataset():
    """
    Menyiapkan dataset dengan membagi data menjadi train, val, dan test,
    dan menyesuaikan ukuran gambar.
    """
    # Buat struktur folder jika belum ada
    if os.path.exists(PROCESSED_DATA_PATH):
        shutil.rmtree(PROCESSED_DATA_PATH)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(PROCESSED_DATA_PATH, split, COVER_IMAGE_FOLDER), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DATA_PATH, split, SECRET_IMAGE_FOLDER), exist_ok=True)

    # --- Proses Gambar Cover ---
    print("Memproses gambar cover...")
    cover_files = [f for f in os.listdir(os.path.join(RAW_DATA_PATH, COVER_IMAGE_FOLDER)) if f.endswith(('.jpg', '.png'))]
    random.shuffle(cover_files) # Acak urutan cover
    
    num_covers = len(cover_files)
    train_end = int(num_covers * TRAIN_SPLIT)
    val_end = train_end + int(num_covers * VAL_SPLIT)

    train_covers = cover_files[:train_end]
    val_covers = cover_files[train_end:val_end]
    test_covers = cover_files[val_end:]
    
    for filename in train_covers:
        resize_and_save(os.path.join(RAW_DATA_PATH, COVER_IMAGE_FOLDER, filename),
                        os.path.join(PROCESSED_DATA_PATH, "train", COVER_IMAGE_FOLDER, filename),
                        COVER_TARGET_SIZE)
    # ... (Tambahkan loop serupa untuk val dan test covers) ...
    for filename in val_covers:
        resize_and_save(os.path.join(RAW_DATA_PATH, COVER_IMAGE_FOLDER, filename),
                        os.path.join(PROCESSED_DATA_PATH, "val", COVER_IMAGE_FOLDER, filename),
                        COVER_TARGET_SIZE)
    for filename in test_covers:
        resize_and_save(os.path.join(RAW_DATA_PATH, COVER_IMAGE_FOLDER, filename),
                        os.path.join(PROCESSED_DATA_PATH, "test", COVER_IMAGE_FOLDER, filename),
                        COVER_TARGET_SIZE)


    # --- Proses Gambar Secret ---
    print("Memproses gambar secret...")
    secret_files = [f for f in os.listdir(os.path.join(RAW_DATA_PATH, SECRET_IMAGE_FOLDER)) if f.endswith(('.jpg', '.png'))]
    
    num_secrets = len(secret_files)
    train_end_secret = int(num_secrets * TRAIN_SPLIT)
    val_end_secret = train_end_secret + int(num_secrets * VAL_SPLIT)
    
    train_secrets = secret_files[:train_end_secret]
    val_secrets = secret_files[train_end_secret:val_end_secret]
    test_secrets = secret_files[val_end_secret:]
    
    for filename in train_secrets:
        resize_and_save(os.path.join(RAW_DATA_PATH, SECRET_IMAGE_FOLDER, filename),
                        os.path.join(PROCESSED_DATA_PATH, "train", SECRET_IMAGE_FOLDER, filename),
                        SECRET_TARGET_SIZE)
    # ... (Tambahkan loop serupa untuk val dan test secrets) ...
    for filename in val_secrets:
        resize_and_save(os.path.join(RAW_DATA_PATH, SECRET_IMAGE_FOLDER, filename),
                        os.path.join(PROCESSED_DATA_PATH, "val", SECRET_IMAGE_FOLDER, filename),
                        SECRET_TARGET_SIZE)
    for filename in test_secrets:
        resize_and_save(os.path.join(RAW_DATA_PATH, SECRET_IMAGE_FOLDER, filename),
                        os.path.join(PROCESSED_DATA_PATH, "test", SECRET_IMAGE_FOLDER, filename),
                        SECRET_TARGET_SIZE)

    print("Dataset telah disiapkan dan disimpan di:", PROCESSED_DATA_PATH)

if __name__ == "__main__":
    prepare_dataset()