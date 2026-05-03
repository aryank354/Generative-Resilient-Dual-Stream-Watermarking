import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Import your GR-DSW Architecture
from gr_dsw.models.vit_autoencoder import WatermarkViTAutoEncoder
from gr_dsw.crypto.hyper_lorenz import generate_chaotic_key, process_watermark
from gr_dsw.watermark.embed import embed_robust_watermark, embed_fragile_watermark
from gr_dsw.watermark.extract import detect_tampering, extract_and_recover
from gr_dsw.utils.metrics import evaluate_quality

class WatermarkAttacks:
    def __init__(self, watermarked_img, base_dir):
        self.img = watermarked_img.copy()
        self.h, self.w = self.img.shape
        self.base_dir = base_dir

    def attack_crop(self, percent):
        attacked = self.img.copy()
        rows = int(self.h * percent)
        attacked[self.h - rows:, :] = 0 
        return attacked

    def attack_collage_splicing(self):
        attacked = self.img.copy()
        # Create a random noise block instead of a flat color
        # A real attacker would splice a real image, which has natural noise
        donor = np.random.randint(0, 256, (80, 80), dtype=np.uint8)
        dh, dw = donor.shape
        start_y, start_x = (self.h - dh) // 2, (self.w - dw) // 2
        attacked[start_y:start_y+dh, start_x:start_x+dw] = donor
        return attacked

    def attack_jpeg(self, quality):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', self.img, encode_param)
        return cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)

    def attack_salt_pepper(self, amount):
        attacked = self.img.copy()
        num_salt = np.ceil(amount * self.img.size * 0.5)
        num_pepper = np.ceil(amount * self.img.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.img.shape]
        attacked[tuple(coords)] = 255
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.img.shape]
        attacked[tuple(coords)] = 0
        return attacked

def compute_tdr(watermarked_img, attacked_img, tamper_map):
    """Accurately computes Tamper Detection Rate based on physical pixel changes"""
    gt_tamper = (np.abs(watermarked_img.astype(np.int32) - attacked_img.astype(np.int32)) > 0)
    detected = (tamper_map == 255)
    total_tampered = np.sum(gt_tamper)
    if total_tampered == 0:
        return 1.0 
    tdr = np.sum(gt_tamper & detected) / total_tampered
    return float(tdr)

def generate_visual_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load Model
    model = WatermarkViTAutoEncoder(latent_dim=256).to(device)
    model_path = os.path.join(base_dir, "gr_dsw", "models", "pretrained_vit.pth")
    if not os.path.exists(model_path):
        print(f"[!] Error: Model not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Paths
    raw_dir = os.path.join(base_dir, "RawImages")
    results_dir = os.path.join(base_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)
    pdf_path = os.path.join(results_dir, "All_Images_Visual_Report.pdf")
    
    raw_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.tiff', '.tif'))]
    if not raw_files:
        print(f"[!] No images found in {raw_dir}")
        return

    secret_key = [1.1, 2.2, 3.3, 4.4]
    chaos_seq = generate_chaotic_key(256, secret_key)

    print(f"[*] Generating Visual PDF Report at {pdf_path}...")

    with PdfPages(pdf_path) as pdf:
        for raw_file in raw_files:
            img_name = os.path.splitext(raw_file)[0]
            print(f"    -> Processing visual grid for: {img_name}")
            
            original_image = cv2.resize(cv2.imread(os.path.join(raw_dir, raw_file), cv2.IMREAD_GRAYSCALE), (256, 256))
            img_tensor = torch.tensor(original_image / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                latent_bits, _ = model(img_tensor)
                
            encrypted_payload = process_watermark(latent_bits.squeeze().cpu().numpy(), chaos_seq)
            
            # QIM Blind Embedding
            robust_img = embed_robust_watermark(original_image, encrypted_payload, delta=16.0)
            watermarked_img = embed_fragile_watermark(robust_img)
            wm_psnr, wm_ssim = evaluate_quality(original_image, watermarked_img)

            attacker = WatermarkAttacks(watermarked_img, base_dir)
            attacks = {
                "Crop 50%": attacker.attack_crop(0.50),
                "Semantic Splicing": attacker.attack_collage_splicing(),
                "JPEG QF=90": attacker.attack_jpeg(90),
                "Salt & Pepper 2%": attacker.attack_salt_pepper(0.02)
            }

            # Prepare the Figure for this image
            num_attacks = len(attacks)
            fig, axes = plt.subplots(num_attacks, 5, figsize=(18, 3.5 * num_attacks))
            plt.subplots_adjust(wspace=0.05, hspace=0.4)
            fig.suptitle(f"GR-DSW Visual Recovery: {img_name}", fontsize=20, fontweight='bold', y=0.98)
            
            cols = ['Original', f'Watermarked\n({wm_psnr:.2f} dB)', 'Attacked', 'Tamper Map (TDR)', 'Recovered (R-PSNR)']
            for ax, col in zip(axes[0], cols):
                ax.set_title(col, fontsize=14, fontweight='bold', pad=15)

            row_idx = 0
            for atk_name, attacked_img in attacks.items():
                
                # Attack Analysis
                tamper_map = detect_tampering(attacked_img)
                tdr = compute_tdr(watermarked_img, attacked_img, tamper_map)
                
                tamper_ratio = np.sum(tamper_map == 255) / tamper_map.size
                num_labels, _ = cv2.connectedComponents(tamper_map)
                is_global_attack = tamper_ratio > 0.85 or num_labels > 50
                
                rec_key = generate_chaotic_key(256, secret_key)
                
                # Blind Extraction
                if is_global_attack:
                    ai_hallucination, _ = extract_and_recover(attacked_img, rec_key, model.decoder, device, tamper_map=np.zeros_like(tamper_map), delta=16.0)
                    final_recovered = ai_hallucination
                    mode_text = "Global Mode"
                else:
                    ai_hallucination, _ = extract_and_recover(attacked_img, rec_key, model.decoder, device, tamper_map=tamper_map, delta=16.0)
                    final_recovered = np.where(tamper_map == 255, ai_hallucination, attacked_img).astype(np.uint8)
                    mode_text = "Local Mode"

                rec_psnr, rec_ssim = evaluate_quality(original_image, final_recovered)

                # --- PLOTTING ---
                axes[row_idx, 0].imshow(original_image, cmap='gray')
                axes[row_idx, 0].set_ylabel(atk_name, fontsize=14, fontweight='bold', labelpad=15)
                
                axes[row_idx, 1].imshow(watermarked_img, cmap='gray')
                
                axes[row_idx, 2].imshow(attacked_img, cmap='gray')
                
                axes[row_idx, 3].imshow(tamper_map, cmap='gray', vmin=0, vmax=255)
                axes[row_idx, 3].set_title(f"TDR: {tdr:.4f}", fontsize=12)
                
                axes[row_idx, 4].imshow(final_recovered, cmap='gray')
                axes[row_idx, 4].set_title(f"{rec_psnr:.2f} dB | {rec_ssim:.3f}\n({mode_text})", fontsize=12)

                for j in range(5):
                    axes[row_idx, j].set_xticks([])
                    axes[row_idx, j].set_yticks([])

                row_idx += 1

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"\n[+] SUCCESS! Multi-page visual report saved to: {pdf_path}")

if __name__ == "__main__":
    generate_visual_report()