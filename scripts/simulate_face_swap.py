import os
import cv2
import numpy as np

def simulate_seamless_faceswap():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "Results")
    attacked_dir = os.path.join(base_dir, "watermarked_attacked_images")
    os.makedirs(attacked_dir, exist_ok=True)

    target_path = os.path.join(results_dir, "Walter_Fresh_Watermarked.png")
    donor_path = os.path.join(base_dir, "RawImages", "Lena.png")

    if not os.path.exists(target_path):
        print(f"[!] Error: Could not find watermarked target at {target_path}")
        return
    if not os.path.exists(donor_path):
        print(f"[!] Error: Could not find donor image at {donor_path}")
        return

    target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    donor_img = cv2.imread(donor_path, cv2.IMREAD_GRAYSCALE)
    
    target_img = cv2.resize(target_img, (256, 256))
    donor_img = cv2.resize(donor_img, (256, 256))

    y1, y2, x1, x2 = 70, 180, 80, 170
    donor_face = donor_img[y1:y2, x1:x2]
    
    mask = np.zeros_like(donor_face)
    cv2.ellipse(mask, (donor_face.shape[1]//2, donor_face.shape[0]//2), 
               (donor_face.shape[1]//2 - 5, donor_face.shape[0]//2 - 5), 
               0, 0, 360, 255, -1)

    target_bgr = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
    donor_face_bgr = cv2.cvtColor(donor_face, cv2.COLOR_GRAY2BGR)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)

    swapped_bgr = cv2.seamlessClone(donor_face_bgr, target_bgr, mask, center, cv2.NORMAL_CLONE)
    swapped_gray = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2GRAY)

    output_filename = "Walter-Cronkite_simswap.png"
    output_path = os.path.join(attacked_dir, output_filename)
    cv2.imwrite(output_path, swapped_gray)

    print(f"\n[+] SUCCESS! Simulated Face Swap saved to: {output_path}")

if __name__ == "__main__":
    simulate_seamless_faceswap()