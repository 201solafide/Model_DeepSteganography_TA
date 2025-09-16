import os
import cv2
from tqdm import tqdm

def resize_image(image_path, size=(64, 64)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    resized = cv2.resize(img, size)
    return resized

def resize_ordered_secret_images(input_dir, output_dir, total_needed=3870):
    os.makedirs(output_dir, exist_ok=True)

    all_images = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))])
    selected = all_images[:total_needed]

    for i, img_name in enumerate(tqdm(selected, desc="Resizing secret images")):
        img_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, f"secret_{i:04d}.png")
        resized = resize_image(img_path)
        if resized is not None:
            cv2.imwrite(output_path, resized)

if __name__ == "__main__":
    input_secret_dir = "data/secret_images/"
    output_secret_dir = "data/secret_resized/"
    total_required = 774 * 5  # 5 secret per frame video
    resize_ordered_secret_images(input_secret_dir, output_secret_dir, total_required)
