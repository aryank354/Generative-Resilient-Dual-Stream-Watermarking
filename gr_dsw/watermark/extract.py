import cv2
import numpy as np
import hashlib
import pywt
import torch
from gr_dsw.crypto.hyper_lorenz import process_watermark

def detect_tampering(attacked_img):
    tamper_map = np.zeros_like(attacked_img)
    
    # 1. Block-wise detection
    for i in range(0, attacked_img.shape[0], 8):
        for j in range(0, attacked_img.shape[1], 8):
            block = attacked_img[i:i+8, j:j+8]
            msb_block = block & 0xFC
            expected_hash = int(hashlib.md5(msb_block.tobytes()).hexdigest()[:2], 16) % 4
            actual_lsb = block & 0x03
            
            if np.any(actual_lsb != expected_hash):
                tamper_map[i:i+8, j:j+8] = 255
                
    # 2. Morphological Closing to fix "Ghosting"
    kernel = np.ones((16, 16), np.uint8)
    solid_tamper_map = cv2.morphologyEx(tamper_map, cv2.MORPH_CLOSE, kernel)
    
    return solid_tamper_map

def extract_and_recover(attacked_channel, orig_cH2, chaotic_key, decoder_model, device, latent_dim=256):
    coeffs = pywt.wavedec2(attacked_channel, 'haar', level=2)
    _, (cH2_atk, _, _), _ = coeffs
    cH2_atk_flat = cH2_atk.flatten()
    
    extracted_repeated = np.zeros(4096, dtype=np.uint8)
    for i in range(4096):
        extracted_repeated[i] = 1 if cH2_atk_flat[i] > orig_cH2[i] else 0
            
    # 16x Majority Voting error correction
    reshaped = extracted_repeated.reshape(16, latent_dim)
    extracted_bits = np.round(np.mean(reshaped, axis=0)).astype(np.uint8)
        
    decrypted_bits = process_watermark(extracted_bits, chaotic_key)
    
    latent_tensor = torch.tensor(decrypted_bits, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        recovered_img_tensor = decoder_model(latent_tensor)
        
    recovered_np = recovered_img_tensor.squeeze().cpu().numpy() * 255.0
    return recovered_np.astype(np.uint8)