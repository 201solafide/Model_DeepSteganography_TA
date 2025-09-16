import cv2
from PIL import Image
import numpy as np

def extract_frame(video_path, frame_number):
    """Mengekstrak satu frame dari video pada posisi tertentu dan mengembalikannya sebagai objek PIL Image."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Atur posisi frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1) # cv2 index-based (0)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Konversi dari BGR (OpenCV) ke RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    return None

def replace_frame(original_video_path, new_frame_pil, frame_number, output_video_path):
    """Membaca video, mengganti satu frame, dan menyimpan sebagai video baru."""
    cap = cv2.VideoCapture(original_video_path)
    
    # Dapatkan properti video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec untuk .mp4
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame == frame_number - 1:
            # Konversi frame PIL baru ke format OpenCV
            new_frame_pil_resized = new_frame_pil.resize((width, height))
            new_frame_cv = cv2.cvtColor(np.array(new_frame_pil_resized), cv2.COLOR_RGB2BGR)
            out.write(new_frame_cv)
        else:
            out.write(frame)
            
        current_frame += 1
        
    cap.release()
    out.release()