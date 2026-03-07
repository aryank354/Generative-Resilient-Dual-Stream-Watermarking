import os
import cv2
import torch
import numpy as np

from gr_dsw.models.vit_autoencoder import WatermarkViTAutoEncoder
from gr_dsw.crypto.hyper_lorenz import generate_chaotic_key, process_watermark
from gr_dsw.watermark.embed import embed_robust_watermark, embed_fragile_watermark

def generate_fresh():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WatermarkViTAutoEncoder(latent_dim=256).to(device)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model.load_state_dict(torch.load(os.path.join(base_dir, "gr_dsw", "models", "pretrained_vit.pth"), map_location=device))
    model.eval()

    raw_path = os.path.join(base_dir, "RawImages", "Walter-Cronkite.tiff")
    original_image = cv2.resize(cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE), (256, 256))

    # 1. AI Compression
    img_tensor = torch.tensor(original_image / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        latent_bits, _ = model(img_tensor)
    
    # 2. Encryption
    secret_key = [1.1, 2.2, 3.3, 4.4]
    chaos_seq = generate_chaotic_key(256, secret_key)
    encrypted_payload = process_watermark(latent_bits.squeeze().cpu().numpy(), chaos_seq)

    # 3. Embedding (Armor increased to 25.0)
    robust_img, _ = embed_robust_watermark(original_image, encrypted_payload, alpha=25.0)
    watermarked_img = embed_fragile_watermark(robust_img)

    # 4. Save Fresh Image
    results_dir = os.path.join(base_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "Walter_Fresh_Watermarked.png")
    cv2.imwrite(output_path, watermarked_img)
    print(f"[+] Fresh watermarked image saved to: {output_path}")

if __name__ == "__main__":
    generate_fresh()