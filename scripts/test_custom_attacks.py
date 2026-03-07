import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pywt

from gr_dsw.models.vit_autoencoder import WatermarkViTAutoEncoder
from gr_dsw.crypto.hyper_lorenz import generate_chaotic_key
from gr_dsw.watermark.extract import detect_tampering, extract_and_recover
from gr_dsw.utils.metrics import evaluate_quality

def process_custom_attacks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WatermarkViTAutoEncoder(latent_dim=256).to(device)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(base_dir, "gr_dsw", "models", "pretrained_vit.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("[!] Error: Pre-trained weights not found.")
        return
    model.eval()

    raw_dir = os.path.join(base_dir, "RawImages")
    attacked_dir = os.path.join(base_dir, "watermarked_attacked_images")
    results_dir = os.path.join(base_dir, "Custom_Attack_Results")
    
    os.makedirs(attacked_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    attacked_files = sorted([f for f in os.listdir(attacked_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))])
    
    if not attacked_files:
        print(f"[!] The folder '{attacked_dir}' is empty. Drop your attacked images here!")
        return

    secret_key = [1.1, 2.2, 3.3, 4.4]

    for atk_file in attacked_files:
        print(f"\n[*] Processing Custom Attack: {atk_file}")
        
        orig_file = None
        for raw in os.listdir(raw_dir):
            if os.path.splitext(raw)[0].lower() in atk_file.lower():
                orig_file = raw
                break
        
        if not orig_file:
            print(f"    [!] Skipping. Could not identify base image name (e.g., 'Lena', 'Jet') in filename.")
            continue
            
        attacked_img = cv2.imread(os.path.join(attacked_dir, atk_file), cv2.IMREAD_GRAYSCALE)
        reference_img = cv2.imread(os.path.join(raw_dir, orig_file), cv2.IMREAD_GRAYSCALE)
        
        attacked_img = cv2.resize(attacked_img, (256, 256))
        reference_img = cv2.resize(reference_img, (256, 256))

        coeffs = pywt.wavedec2(reference_img, 'haar', level=2)
        _, (cH2_orig, _, _), _ = coeffs
        orig_cH2_flat = cH2_orig.flatten()

        tamper_map = detect_tampering(attacked_img)
        tamper_ratio = np.sum(tamper_map == 255) / tamper_map.size

        if tamper_ratio > 0.40:
            print("    -> Catastrophic Damage Detected. Bypassing AI.")
            final_recovered = cv2.medianBlur(attacked_img, 3)
            tamper_map.fill(255)
        else:
            print("    -> Localized Attack Detected. AI Hallucinating missing pixels...")
            receiver_key = generate_chaotic_key(256, secret_key)
            # PASSING THE TAMPER MAP
            ai_hallucination = extract_and_recover(attacked_img, orig_cH2_flat, receiver_key, model.decoder, device, tamper_map=tamper_map)
            final_recovered = np.where(tamper_map == 255, ai_hallucination, attacked_img).astype(np.uint8)

        rec_psnr, rec_ssim = evaluate_quality(reference_img, final_recovered)
        print(f"    -> Recovery Success: RPSNR = {rec_psnr:.2f} dB")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"Custom Attack: {atk_file} | RPSNR: {rec_psnr:.2f} dB", fontsize=14, fontweight='bold')
        
        axes[0].imshow(attacked_img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title("Your Custom Attack")
        axes[0].axis('off')
        
        axes[1].imshow(tamper_map, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title("Detected Tamper Map")
        axes[1].axis('off')
        
        axes[2].imshow(final_recovered, cmap='gray', vmin=0, vmax=255)
        axes[2].set_title(f"AI Recovered (RPSNR: {rec_psnr:.1f})")
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(results_dir, f"CustomResult_{os.path.splitext(atk_file)[0]}.png")
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    process_custom_attacks()