# app/decoder_backend.py

import torch
from torchvision import transforms
from PIL import Image
import os
from models.stego_encoder_decoder import StegoEncoderDecoder
from utils.video_utils import extract_frame

def process_decoding(stego_video_path, frame_position, num_secrets):
    if not 1 <= num_secrets <= 5:
        raise ValueError("Jumlah gambar rahasia harus antara 1 dan 5.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Path ini sudah benar karena dijalankan dari root folder DEEP-STEGA
    MODEL_PATH = f"checkpoints/model_{num_secrets}secrets_best_extended.pth"
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint tidak ditemukan: {MODEL_PATH}")
        
    model = StegoEncoderDecoder(num_secrets=num_secrets).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Ekstrak stego frame dari video
    stego_frame_pil = extract_frame(stego_video_path, frame_position)
    if stego_frame_pil is None:
        raise ValueError(f"Frame pada posisi {frame_position} tidak dapat diekstrak.")

    # Pra-pemrosesan stego frame
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    stego_tensor = transform(stego_frame_pil).unsqueeze(0).to(device)

    # Jalankan proses decoding
    with torch.no_grad():
        _, revealed_tensors = model(stego_tensor, torch.zeros(1, num_secrets, 3, 64, 64).to(device)) # Beri input dummy untuk secrets

    # Simpan gambar hasil ekstraksi
    output_image_filenames = []
    results_dir = os.path.join(os.path.dirname(__file__), 'static', 'results') # Definisikan path absolut
    
    for i, tensor in enumerate(revealed_tensors):
        revealed_img = transforms.ToPILImage()(tensor.cpu())
        filename = f"revealed_{frame_position}_{i+1}.png"
        save_path = os.path.join(results_dir, filename)
        revealed_img.save(save_path)
        output_image_filenames.append(filename)
        
    return output_image_filenames