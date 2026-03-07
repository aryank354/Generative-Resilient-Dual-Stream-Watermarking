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

    # ==========================================
    # 1. SPATIAL TAMPERING & FORGERY (Localized)
    # ==========================================
    def attack_crop(self, percent):
        attacked = self.img.copy()
        rows = int(self.h * percent)
        attacked[self.h - rows:, :] = 0 
        return attacked

    def attack_row_tampering(self, percent):
        attacked = self.img.copy()
        rows = int(self.h * percent)
        start = (self.h - rows) // 2
        attacked[start:start + rows, :] = 0
        return attacked

    def attack_column_tampering(self, percent):
        attacked = self.img.copy()
        cols = int(self.w * percent)
        start = (self.w - cols) // 2
        attacked[:, start:start + cols] = 0
        return attacked

    def attack_grid_tampering(self, grid_size=4):
        attacked = self.img.copy()
        step_h, step_w = self.h // grid_size, self.w // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                if (i + j) % 2 == 0:  
                    attacked[i*step_h:(i+1)*step_h, j*step_w:(j+1)*step_w] = 0
        return attacked

    def attack_content_removal(self, box_size=80):
        attacked = self.img.copy()
        start_y, start_x = (self.h - box_size) // 2, (self.w - box_size) // 2
        attacked[start_y:start_y+box_size, start_x:start_x+box_size] = 0
        return attacked

    def attack_text_insertion(self, text="TAMPERED"):
        attacked = self.img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(attacked, text, (self.w // 6, self.h // 2), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        return attacked

    def attack_collage_splicing(self):
        attacked = self.img.copy()
        donor_path = os.path.join(self.base_dir, "RawImages", "Lena.png")
        if not os.path.exists(donor_path):
            donor = np.zeros((80, 80), dtype=np.uint8)
            donor.fill(200) 
        else:
            donor = cv2.resize(cv2.imread(donor_path, cv2.IMREAD_GRAYSCALE), (80, 80))
        
        dh, dw = donor.shape
        start_y, start_x = (self.h - dh) // 2, (self.w - dw) // 2
        attacked[start_y:start_y+dh, start_x:start_x+dw] = donor
        return attacked

    # ==========================================
    # 2. SIGNAL DEGRADATION (COMPRESSION & NOISE)
    # ==========================================
    def attack_jpeg(self, quality):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', self.img, encode_param)
        return cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)

    def attack_gaussian_noise(self, var):
        sigma = var ** 0.5
        gaussian = np.random.normal(0, sigma * 255, (self.h, self.w))
        return np.clip(self.img + gaussian, 0, 255).astype(np.uint8)

    def attack_salt_pepper(self, amount):
        attacked = self.img.copy()
        num_salt = np.ceil(amount * self.img.size * 0.5)
        num_pepper = np.ceil(amount * self.img.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.img.shape]
        attacked[tuple(coords)] = 255
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.img.shape]
        attacked[tuple(coords)] = 0
        return attacked

    def attack_speckle_noise(self, var):
        gauss = np.random.normal(0, var ** 0.5, (self.h, self.w))
        return np.clip(self.img + self.img * gauss, 0, 255).astype(np.uint8)

    # ==========================================
    # 3. FILTERING & ENHANCEMENT
    # ==========================================
    def attack_median_filter(self, ksize):
        return cv2.medianBlur(self.img, ksize)

    def attack_gaussian_lpf(self, ksize):
        return cv2.GaussianBlur(self.img, (ksize, ksize), 0)

    def attack_sharpening(self):
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        return np.clip(cv2.filter2D(self.img, -1, kernel), 0, 255).astype(np.uint8)

    def attack_histogram_equalization(self):
        return cv2.equalizeHist(self.img)

    def attack_motion_blur(self, size=5):
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(self.img, -1, kernel)


# ==========================================
# 4. COMPREHENSIVE PIPELINE EXECUTION
# ==========================================
def run_comprehensive_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WatermarkViTAutoEncoder(latent_dim=256).to(device)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model.load_state_dict(torch.load(os.path.join(base_dir, "gr_dsw", "models", "pretrained_vit.pth"), map_location=device))
    model.eval()

    raw_dir = os.path.join(base_dir, "RawImages")
    results_dir = os.path.join(base_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)
    pdf_path = os.path.join(results_dir, "GR_DSW_Comprehensive_Attacks_Report.pdf")

    secret_key = [1.1, 2.2, 3.3, 4.4]
    chaos_seq = generate_chaotic_key(256, secret_key)
    raw_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.tiff', '.tif'))]

    if not raw_files:
        print(f"[!] No images found in {raw_dir}")
        return

    with PdfPages(pdf_path) as pdf:
        for raw_file in raw_files:
            img_name = os.path.splitext(raw_file)[0]
            print(f"\n[*] Processing Image: {img_name}")
            
            # Load & Embed
            original_image = cv2.resize(cv2.imread(os.path.join(raw_dir, raw_file), cv2.IMREAD_GRAYSCALE), (256, 256))
            img_tensor = torch.tensor(original_image / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                latent_bits, _ = model(img_tensor)
            
            encrypted_payload = process_watermark(latent_bits.squeeze().cpu().numpy(), chaos_seq)
            robust_img, orig_cH2_flat = embed_robust_watermark(original_image, encrypted_payload, alpha=8.0)
            watermarked_img = embed_fragile_watermark(robust_img)
            wm_psnr, _ = evaluate_quality(original_image, watermarked_img)

            attacker = WatermarkAttacks(watermarked_img, base_dir)
            attacks = {
                "Crop 10%": attacker.attack_crop(0.10),
                "Crop 30%": attacker.attack_crop(0.30),
                "Crop 50%": attacker.attack_crop(0.50),
                "Row Tamper 25%": attacker.attack_row_tampering(0.25),
                "Row Tamper 50%": attacker.attack_row_tampering(0.50),
                "Col Tamper 25%": attacker.attack_column_tampering(0.25),
                "Grid Tamper 4x4": attacker.attack_grid_tampering(4),
                "Content Removal (Hole)": attacker.attack_content_removal(80),
                "Semantic Splicing": attacker.attack_collage_splicing(),
                "Text Insertion": attacker.attack_text_insertion("TAMPERED"),
                "JPEG QF=90": attacker.attack_jpeg(90),
                "JPEG QF=70": attacker.attack_jpeg(70),
                "JPEG QF=50": attacker.attack_jpeg(50),
                "Gaussian Noise v=0.01": attacker.attack_gaussian_noise(0.01),
                "Gaussian Noise v=0.05": attacker.attack_gaussian_noise(0.05),
                "Salt & Pepper 2%": attacker.attack_salt_pepper(0.02),
                "Speckle Noise 4%": attacker.attack_speckle_noise(0.04),
                "Median Filter 3x3": attacker.attack_median_filter(3),
                "Gaussian LPF 3x3": attacker.attack_gaussian_lpf(3),
                "Sharpening": attacker.attack_sharpening(),
                "Hist Equalization": attacker.attack_histogram_equalization(),
                "Motion Blur 5x5": attacker.attack_motion_blur(5)
            }

            for atk_name, attacked_img in attacks.items():
                print(f"    -> Evaluating: {atk_name}")
                
                tamper_map = detect_tampering(attacked_img)
                tamper_ratio = np.sum(tamper_map == 255) / tamper_map.size
                num_labels, _ = cv2.connectedComponents(tamper_map)
                
                # =========================================================
                # SOTA UPGRADE: THE SMART CIRCUIT BREAKER
                # =========================================================
                is_global_attack = tamper_ratio > 0.85 or num_labels > 50
                
                if is_global_attack:
                    # Global Attack -> Bypass AI, apply baseline filter to survive
                    # This ensures the graceful degradation to ~28 dB PSNR!
                    final_recovered = cv2.medianBlur(attacked_img, 3)
                    routing_msg = "Global Attack -> Bypassed AI"
                else:
                    # Localized Attack -> Extract background memory and let AI inpaint
                    receiver_key = generate_chaotic_key(256, secret_key)
                    
                    ai_hallucination = extract_and_recover(attacked_img, orig_cH2_flat, receiver_key, model.decoder, device, tamper_map=tamper_map)
                    
                    # Stitch the AI hallucination ONLY into the white holes
                    final_recovered = np.where(tamper_map == 255, ai_hallucination, attacked_img).astype(np.uint8)
                    routing_msg = "Localized Attack -> AI Inpainting"

                rec_psnr, rec_ssim = evaluate_quality(original_image, final_recovered)

                # Plotting for PDF
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                fig.suptitle(f"{img_name} | Attack: {atk_name} | Embed PSNR: {wm_psnr:.1f} dB | Rec PSNR: {rec_psnr:.2f} dB", fontsize=14, fontweight='bold')
                
                axes[0].imshow(original_image, cmap='gray')
                axes[0].set_title("Original")
                axes[0].axis('off')

                axes[1].imshow(attacked_img, cmap='gray')
                axes[1].set_title("Attacked Image")
                axes[1].axis('off')

                # Using vmin/vmax to guarantee 100% white isn't rendered as black
                axes[2].imshow(tamper_map, cmap='gray', vmin=0, vmax=255)
                axes[2].set_title(f"Tamper Map ({tamper_ratio*100:.1f}% | {num_labels} clusters)")
                axes[2].axis('off')

                axes[3].imshow(final_recovered, cmap='gray')
                axes[3].set_title(f"{routing_msg}\nSSIM: {rec_ssim:.3f}")
                axes[3].axis('off')

                plt.tight_layout()
                pdf.savefig(fig)  
                plt.close()

    print(f"\n[+] SUCCESS! Massive Comprehensive PDF generated at: {pdf_path}")

if __name__ == "__main__":
    run_comprehensive_evaluation()