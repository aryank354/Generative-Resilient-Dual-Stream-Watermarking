import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from gr_dsw.models.vit_autoencoder import WatermarkViTAutoEncoder
from gr_dsw.crypto.hyper_lorenz import generate_chaotic_key, process_watermark
from gr_dsw.watermark.embed import embed_robust_watermark, embed_fragile_watermark
from gr_dsw.watermark.extract import detect_tampering, extract_and_recover
from gr_dsw.utils.metrics import evaluate_quality


# ===========================================================
# ATTACK CLASS
# ===========================================================
class WatermarkAttacks:
    def __init__(self, watermarked_img, base_dir, all_raw_images):
        self.img = watermarked_img.copy()
        self.h, self.w = self.img.shape
        self.base_dir = base_dir
        # Pool of other images to use as real splice donors
        self.donor_pool = all_raw_images

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

    def attack_content_removal(self, box_size):
        attacked = self.img.copy()
        start_y = max(0, (self.h - box_size) // 2)
        start_x = max(0, (self.w - box_size) // 2)
        end_y = min(self.h, start_y + box_size)
        end_x = min(self.w, start_x + box_size)
        attacked[start_y:end_y, start_x:end_x] = 0
        return attacked

    def attack_text_insertion(self, text="TAMPERED"):
        attacked = self.img.copy()
        cv2.putText(attacked, text, (self.w // 6, self.h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        return attacked

    def attack_collage_splicing(self, host_name="Peppers"):
        """
        FIX: Use a real image from the dataset pool as the donor.
        Picks the first image that is NOT the host image.
        """
        attacked = self.img.copy()
        donor = None
        for name, img_arr in self.donor_pool.items():
            # Use the first image that is NOT the host
            if host_name.lower() not in name.lower():
                donor = cv2.resize(img_arr, (80, 80))
                donor_name = name
                break

        if donor is None:
            # Ultimate fallback: use a rotated version of the host itself
            donor = cv2.resize(np.rot90(self.img, 2), (80, 80))
            donor_name = "rotated host"

        print(f"      [Splicing] Using '{donor_name}' as real semantic donor.")
        dh, dw = donor.shape
        start_y = (self.h - dh) // 2
        start_x = (self.w - dw) // 2
        attacked[start_y:start_y + dh, start_x:start_x + dw] = donor
        return attacked

    def attack_jpeg(self, quality):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', self.img, encode_param)
        return cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)

    def attack_salt_pepper(self, amount):
        attacked = self.img.copy()
        num_salt = int(np.ceil(amount * self.img.size * 0.5))
        num_pepper = int(np.ceil(amount * self.img.size * 0.5))
        # Salt
        coords = [np.random.randint(0, i, num_salt) for i in self.img.shape]
        attacked[tuple(coords)] = 255
        # Pepper
        coords = [np.random.randint(0, i, num_pepper) for i in self.img.shape]
        attacked[tuple(coords)] = 0
        return attacked

    def attack_motion_blur(self, size=5):
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size) / size
        return cv2.filter2D(self.img, -1, kernel)


# ===========================================================
# TDR COMPUTATION  (THE FIX)
# ===========================================================
def compute_tdr(watermarked_img, attacked_img, tamper_map):
    """
    Computes the true Tamper Detection Rate.

    ground_truth tampered pixels: any pixel that changed between
    watermarked and attacked image.
    detected tampered pixels: pixels flagged by the LSB hash map.

    TDR = |GT_tampered ∩ Detected| / |GT_tampered|

    Returns 1.0 if the image was not tampered at all (trivially correct).
    """
    gt_tamper = (np.abs(
        watermarked_img.astype(np.int32) - attacked_img.astype(np.int32)
    ) > 0)

    detected = (tamper_map == 255)

    total_tampered = np.sum(gt_tamper)
    if total_tampered == 0:
        return 1.0  # No tampering occurred — perfect by definition

    tdr = np.sum(gt_tamper & detected) / total_tampered
    return float(tdr)


# ===========================================================
# MAIN GRID GENERATOR
# ===========================================================
def generate_peppers_grid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── 1. Load Model ──────────────────────────────────────
    model = WatermarkViTAutoEncoder(latent_dim=256).to(device)
    model.load_state_dict(torch.load(
        os.path.join(base_dir, "gr_dsw", "models", "pretrained_vit.pth"),
        map_location=device
    ))
    model.eval()

    # ── 2. Load ALL raw images for donor pool ───────────────
    raw_dir = os.path.join(base_dir, "RawImages")
    results_dir = os.path.join(base_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)

    all_raw_images = {}
    for fname in os.listdir(raw_dir):
        if fname.lower().endswith(('.png', '.jpg', '.tiff', '.tif')):
            img = cv2.imread(os.path.join(raw_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                all_raw_images[os.path.splitext(fname)[0]] = \
                    cv2.resize(img, (256, 256))

    if not all_raw_images:
        print(f"[!] No images found in {raw_dir}")
        return

    # ── 3. Select host image (Peppers) ─────────────────────
    host_key = None
    for k in all_raw_images:
        if "pepper" in k.lower():
            host_key = k
            break
    if host_key is None:
        host_key = list(all_raw_images.keys())[0]
        print(f"[!] Peppers not found, using '{host_key}' as host.")

    original_image = all_raw_images[host_key]
    print(f"[*] Host image: {host_key}")

    # ── 4. Embed Watermark ──────────────────────────────────
    secret_key = [1.1, 2.2, 3.3, 4.4]
    chaos_seq = generate_chaotic_key(256, secret_key)

    img_tensor = torch.tensor(
        original_image / 255.0, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        latent_bits, _ = model(img_tensor)

    encrypted_payload = process_watermark(
        latent_bits.squeeze().cpu().numpy(), chaos_seq
    )
    robust_img, orig_cH2_flat = embed_robust_watermark(
        original_image, encrypted_payload, alpha=8.0
    )
    watermarked_img = embed_fragile_watermark(robust_img)

    wm_psnr, wm_ssim = evaluate_quality(original_image, watermarked_img)
    print(f"[*] Watermarked PSNR: {wm_psnr:.2f} dB | SSIM: {wm_ssim:.4f}")

    # ── 5. Define Attacks ───────────────────────────────────
    attacker = WatermarkAttacks(watermarked_img, base_dir, all_raw_images)

    attacks = {
        "Content Removal\n(80×80 box)":
            attacker.attack_content_removal(80),
        "Semantic Splicing\n(real donor patch)":
            attacker.attack_collage_splicing(host_name=host_key),
        "Text Insertion\n(\"TAMPERED\")":
            attacker.attack_text_insertion("TAMPERED"),
        "Crop 50%":
            attacker.attack_crop(0.50),
        "Row Tamper 50%":
            attacker.attack_row_tampering(0.50),
        "JPEG QF=90":
            attacker.attack_jpeg(90),
        "Salt & Pepper 2%":
            attacker.attack_salt_pepper(0.02),
        "Motion Blur 5×5":
            attacker.attack_motion_blur(5),
    }

    # ── 6. Build Figure ─────────────────────────────────────
    num_attacks = len(attacks)
    fig, axes = plt.subplots(
        num_attacks, 5,
        figsize=(20, 3.2 * num_attacks)
    )
    plt.subplots_adjust(wspace=0.04, hspace=0.35)

    col_titles = [
        "Original",
        f"Watermarked\n({wm_psnr:.2f} dB | SSIM {wm_ssim:.4f})",
        "Attacked",
        "Tamper Map\n(Computed TDR)",       # ← explicit label
        "Recovered\n(Rec. PSNR | SSIM)"
    ]
    for ax, col in zip(axes[0], col_titles):
        ax.set_title(col, fontsize=13, fontweight='bold', pad=10)

    # ── 7. Process Each Attack ──────────────────────────────
    for row_idx, (atk_name, attacked_img) in enumerate(attacks.items()):
        print(f"\n  [Attack] {atk_name.replace(chr(10), ' ')}")

        # Tamper detection
        tamper_map = detect_tampering(attacked_img)

        # ── REAL TDR COMPUTATION (THE FIX) ──────────────────
        tdr = compute_tdr(watermarked_img, attacked_img, tamper_map)
        print(f"    TDR (computed): {tdr:.4f}")

        # Circuit breaker decision
        tamper_ratio = np.sum(tamper_map == 255) / tamper_map.size
        num_labels, _ = cv2.connectedComponents(tamper_map)
        is_global = tamper_ratio > 0.85 or num_labels > 50

        receiver_key = generate_chaotic_key(256, secret_key)

        if is_global:
            ai_hallucination = extract_and_recover(
                attacked_img, orig_cH2_flat, receiver_key,
                model.decoder, device,
                tamper_map=np.zeros_like(tamper_map)
            )
            final_recovered = ai_hallucination.astype(np.uint8)
            mode_text = "Global Mode"
        else:
            ai_hallucination = extract_and_recover(
                attacked_img, orig_cH2_flat, receiver_key,
                model.decoder, device,
                tamper_map=tamper_map
            )
            final_recovered = np.where(
                tamper_map == 255,
                ai_hallucination,
                attacked_img
            ).astype(np.uint8)
            mode_text = "Local Mode"

        rec_psnr, rec_ssim = evaluate_quality(original_image, final_recovered)
        print(f"    Rec PSNR: {rec_psnr:.2f} dB | SSIM: {rec_ssim:.4f} | {mode_text}")

        # ── Plot Row ─────────────────────────────────────────
        # Col 0: Original
        axes[row_idx, 0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
        axes[row_idx, 0].set_ylabel(
            atk_name, fontsize=11, fontweight='bold', labelpad=10
        )

        # Col 1: Watermarked
        axes[row_idx, 1].imshow(watermarked_img, cmap='gray', vmin=0, vmax=255)

        # Col 2: Attacked
        axes[row_idx, 2].imshow(attacked_img, cmap='gray', vmin=0, vmax=255)

        # Col 3: Tamper Map — with REAL computed TDR
        axes[row_idx, 3].imshow(tamper_map, cmap='gray', vmin=0, vmax=255)
        tdr_color = 'green' if tdr >= 0.99 else ('orange' if tdr >= 0.90 else 'red')
        axes[row_idx, 3].set_title(
            f"TDR: {tdr:.4f}",    # ← REAL value, not hardcoded
            fontsize=11,
            color=tdr_color,
            fontweight='bold'
        )

        # Col 4: Recovered
        axes[row_idx, 4].imshow(final_recovered, cmap='gray', vmin=0, vmax=255)
        axes[row_idx, 4].set_title(
            f"{rec_psnr:.2f} dB | {rec_ssim:.4f}\n({mode_text})",
            fontsize=11
        )

        # Clean axes
        for j in range(5):
            axes[row_idx, j].set_xticks([])
            axes[row_idx, j].set_yticks([])

    # ── 8. Figure-level annotation ──────────────────────────
    fig.suptitle(
        f"GR-DSW Visual Recovery Grid — Host: {host_key} "
        f"| Embedding: α=8.0 | Extraction: Non-Blind\n"
        f"TDR values are computed from ground-truth pixel difference maps.",
        fontsize=12,
        y=1.01
    )

    # ── 9. Save ─────────────────────────────────────────────
    png_path = os.path.join(results_dir, "Peppers_Visual_Grid.png")
    pdf_path = os.path.join(results_dir, "Peppers_Visual_Grid.pdf")

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[+] Grid saved:")
    print(f"    PNG → {png_path}")
    print(f"    PDF → {pdf_path}")
    print("[+] All TDR values are computed — none are hardcoded.")


if __name__ == "__main__":
    generate_peppers_grid()