# import os
# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import torchvision.transforms as transforms

# class StegoDataset(Dataset):
#     def __init__(self, frame_dir, secret_dir, transform=None):
#         self.frame_dir = frame_dir
#         self.secret_dir = secret_dir

#         # Gunakan transformasi default jika tidak diberikan
#         if not callable(transform):
#             self.transform = transforms.Compose([
#                 transforms.Resize((64, 64)),  # pastikan semua ukuran seragam
#                 transforms.ToTensor()         # Normalisasi [0, 1]
#             ])
#         else:
#             self.transform = transform

#         self.frame_ids = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])

#     def __len__(self):
#         return len(self.frame_ids)

#     def __getitem__(self, idx):
#         # Load cover frame
#         frame_path = os.path.join(self.frame_dir, self.frame_ids[idx])
#         cover = Image.open(frame_path).convert('RGB')
#         cover = self.transform(cover)  # [3, 64, 64]

#         # Load 5 secret images
#         secrets = []
#         base_id = idx * 5
#         for i in range(5):
#             secret_path = os.path.join(self.secret_dir, f"secret_{base_id + i:04d}.png")
#             secret = Image.open(secret_path).convert('RGB')
#             secret = self.transform(secret)  # [3, 64, 64]
#             secrets.append(secret)

#         # Gabungkan 5 secret images: [3*5=15, 64, 64]
#         secret_concat = torch.cat(secrets, dim=0)

#         return cover, secret_concat
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SteganographyDataset(Dataset):
    """
    Dataset kustom untuk memuat gambar cover dan secret secara acak
    dari folder.
    """
    def __init__(self, data_dir, num_secrets, transform=None):
        """
        Inisialisasi dataset.
        Args:
            data_dir (str): Path ke folder data (misal: 'data/processed/train').
            num_secrets (int): Jumlah secret images yang akan dimuat per cover.
            transform (callable, optional): Transformasi yang akan diterapkan pada gambar.
        """
        self.cover_dir = os.path.join(data_dir, "covers")
        self.secret_dir = os.path.join(data_dir, "secrets")
        self.num_secrets = num_secrets
        self.transform = transform
        
        self.cover_files = [f for f in os.listdir(self.cover_dir) if f.endswith(('.jpg', '.png'))]
        self.secret_files = [f for f in os.listdir(self.secret_dir) if f.endswith(('.jpg', '.png'))]
        
        if not self.cover_files:
            raise FileNotFoundError(f"Tidak ada gambar cover ditemukan di {self.cover_dir}")
        if not self.secret_files:
            raise FileNotFoundError(f"Tidak ada gambar secret ditemukan di {self.secret_dir}")

    def __len__(self):
        # Jumlah sampel adalah jumlah cover images
        return len(self.cover_files)

    def __getitem__(self, idx):
        # 1. Pilih cover image
        cover_path = os.path.join(self.cover_dir, self.cover_files[idx])
        cover_image = Image.open(cover_path).convert("RGB")
        
        # 2. Pilih N secret images secara acak
        secret_images = []
        secret_indices = random.sample(range(len(self.secret_files)), self.num_secrets)
        for i in secret_indices:
            secret_path = os.path.join(self.secret_dir, self.secret_files[i])
            secret_image = Image.open(secret_path).convert("RGB")
            secret_images.append(secret_image)
            
        # 3. Terapkan transformasi jika ada
        if self.transform:
            cover_image = self.transform(cover_image)
            secret_images = [self.transform(img) for img in secret_images]
            
        # Kembalikan cover dan list of secret images
        return cover_image, secret_images