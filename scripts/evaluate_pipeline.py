import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from gr_dsw.models.vit_autoencoder import WatermarkViTAutoEncoder
from gr_dsw.crypto.hyper_lorenz import generate_chaotic_key, process_watermark
from gr_dsw.watermark.embed import embed_robust_watermark, embed_fragile_watermark
from gr_dsw.watermark.extract import detect_tampering, extract_and_recover
from gr_dsw.utils.metrics import evaluate_quality
import attacks

def run_comprehensive_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize with the 256-bit hyper-redundant brain
    model = WatermarkViTAutoEncoder(latent_dim=256).to(device) 
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(base_dir, "gr_dsw", "models", "pretrained_vit.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("[!] Pre-trained weights not found. Run train_vit.py first!")
        return
    model.eval()

    raw_images_dir = os.path.join(base_dir, "RawImages")
    results_dir = os.path.join(base_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)
    pdf_path = os.path.join(results_dir, "GR_DSW_Comprehensive_Results.pdf")

    image_files = sorted([f for f in os.listdir(raw_images_dir) if f.lower().endswith(('.png', '.tiff', '.tif', '.jpg', '.jpeg'))])
    
    # Define the 6 attack scenarios
    attack_funcs = {
        "Crop Attack": attacks.attack_crop,
        "Text Insertion": attacks.attack_text_insertion,
        "Face Swap Sim": attacks.attack_face_swap_sim,
        "JPEG Compress (Q=60)": lambda img: attacks.attack_jpeg_compression(img, 60),
        "Gaussian Noise": attacks.attack_noise_gaussian,
        "Salt & Pepper": attacks.attack_noise_sp
    }

    # Secret cryptographic key
    secret_key = [1.1, 2.2, 3.3, 4.4]

    with PdfPages(pdf_path) as pdf:
        for img_name in image_files:
            print(f"\n[*] Processing: {img_name}")
            img_path = os.path.join(raw_images_dir, img_name)
            original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if original_image is None: 
                continue
            original_image = cv2.resize(original_image, (256, 256))
            
            img_tensor = torch.tensor(original_image / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # 1. Extract & Encrypt
            with torch.no_grad(): 
                latent_bits, _ = model(img_tensor)
            chaos_seq = generate_chaotic_key(256, secret_key) 
            encrypted_payload = process_watermark(latent_bits.squeeze().cpu().numpy(), chaos_seq)

            # 2. Embed (alpha=2.0 for high imperceptibility)
            robust_img, orig_S = embed_robust_watermark(original_image, encrypted_payload, alpha=2.0)
            watermarked_img = embed_fragile_watermark(robust_img)
            psnr_emb, ssim_emb = evaluate_quality(original_image, watermarked_img)

            # 3. Plotting Setup
            fig, axes = plt.subplots(len(attack_funcs), 5, figsize=(20, 4 * len(attack_funcs)))
            fig.suptitle(f"GR-DSW Robustness Analysis: {img_name}\nEmbedding PSNR: {psnr_emb:.2f} dB, SSIM: {ssim_emb:.4f}", fontsize=20, fontweight='bold')
            
            row_idx = 0
            for attack_name, atk_fn in attack_funcs.items():
                print(f"    -> Running {attack_name}...")
                
                # Apply the specific attack
                attacked_img = atk_fn(watermarked_img)
                tamper_map = detect_tampering(attacked_img)
                
                # Check for Global vs Local attack (>40% damage)
                tamper_ratio = np.sum(tamper_map == 255) / tamper_map.size
                
                if tamper_ratio > 0.40:
                    # Global Attack Bypass (Noise/JPEG)
                    final_recovered = cv2.medianBlur(attacked_img, 3)
                    # Flag the whole map as attacked for the PDF
                    tamper_map.fill(255) 
                else:
                    # Local Attack Healing (Crop, Text, Forgery)
                    receiver_key = generate_chaotic_key(256, secret_key) 
                    ai_hallucination = extract_and_recover(attacked_img, orig_S, receiver_key, model.decoder, device)
                    # Splice perfectly using the solid morphological tamper map
                    final_recovered = np.where(tamper_map == 255, ai_hallucination, attacked_img).astype(np.uint8)
                
                # Evaluate final recovery
                rec_psnr, rec_ssim = evaluate_quality(original_image, final_recovered)

                # --- PLOTTING ROW (with vmin=0, vmax=255 to fix black maps) ---
                ax = axes[row_idx]
                ax[0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
                ax[0].set_title("Original Image")
                ax[0].axis('off')

                ax[1].imshow(watermarked_img, cmap='gray', vmin=0, vmax=255)
                ax[1].set_title(f"Watermarked\nPSNR: {psnr_emb:.1f} dB")
                ax[1].axis('off')

                ax[2].imshow(attacked_img, cmap='gray', vmin=0, vmax=255)
                ax[2].set_title(f"Attacked: {attack_name}")
                ax[2].axis('off')

                ax[3].imshow(tamper_map, cmap='gray', vmin=0, vmax=255)
                ax[3].set_title("Tamper Detection Map")
                ax[3].axis('off')

                ax[4].imshow(final_recovered, cmap='gray', vmin=0, vmax=255)
                ax[4].set_title(f"Recovered\nPSNR: {rec_psnr:.1f} dB, SSIM: {rec_ssim:.3f}")
                ax[4].axis('off')

                row_idx += 1

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            pdf.savefig(fig)
            plt.close()

    print(f"\n[+] SUCCESS! Comprehensive PDF report saved to: {pdf_path}")

if __name__ == "__main__":
    run_comprehensive_evaluation()