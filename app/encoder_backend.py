import torch
import cv2
from torchvision import transforms
from PIL import Image
import os
from models.stego_encoder_decoder import StegoEncoderDecoder
from utils.video_utils import extract_frame, replace_frame

# Impor dengan path yang benar relatif terhadap root proyek
from models.stego_encoder_decoder import StegoEncoderDecoder
from utils.video_utils import extract_frame, replace_frame

def process_encoding(video_path, secret_image_paths, frame_position):
    num_secrets = len(secret_image_paths)
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

    # 2. Ekstrak frame dari video
    cover_frame_pil = extract_frame(video_path, frame_position)
    if cover_frame_pil is None:
        raise ValueError(f"Frame pada posisi {frame_position} tidak dapat diekstrak.")

    # 3. Pra-pemrosesan gambar
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    secret_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    cover_tensor = transform(cover_frame_pil).unsqueeze(0).to(device)
    
    secrets_list = []
    for img_path in secret_image_paths:
        secret_img = Image.open(img_path).convert("RGB")
        secrets_list.append(secret_transform(secret_img))
    
    secrets_tensor = torch.stack(secrets_list, dim=1).to(device)

    # 4. Jalankan proses encoding
    with torch.no_grad():
        stego_tensor = model.encoder(cover_tensor, secrets_tensor)

    # 5. Pasca-pemrosesan stego frame
    stego_frame_pil = transforms.ToPILImage()(stego_tensor.squeeze(0).cpu())

    # 6. Gantikan frame di video dan simpan video baru
    output_video_filename = f"stego_{os.path.basename(video_path)}"
    output_video_path = os.path.join("app", "static", "results", output_video_filename)
    
    replace_frame(video_path, stego_frame_pil, frame_position, output_video_path)
    
    return output_video_filename