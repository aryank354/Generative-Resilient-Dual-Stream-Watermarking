import cv2
import numpy as np
import hashlib
import pywt
import torch
from gr_dsw.crypto.hyper_lorenz import process_watermark

def detect_tampering(attacked_img):
    tamper_map = np.zeros_like(attacked_img)
    
    for i in range(0, attacked_img.shape[0], 8):
        for j in range(0, attacked_img.shape[1], 8):
            block = attacked_img[i:i+8, j:j+8]
            msb_block = block & 0xFC
            
            # UPGRADED: SHA-256 instead of MD5
            expected_hash = int(hashlib.sha256(msb_block.tobytes()).hexdigest()[:2], 16) % 4
            actual_lsb = block & 0x03
            
            if np.any(actual_lsb != expected_hash):
                tamper_map[i:i+8, j:j+8] = 255
                
    kernel_close = np.ones((16, 16), np.uint8)
    solid_tamper_map = cv2.morphologyEx(tamper_map, cv2.MORPH_CLOSE, kernel_close)
    kernel_dilate = np.ones((12, 12), np.uint8)
    return cv2.dilate(solid_tamper_map, kernel_dilate, iterations=1)

def extract_and_recover(attacked_channel, chaotic_key, decoder_model, device, tamper_map=None, latent_dim=256, delta=16.0):
    # Notice orig_cH2 is GONE from the arguments!
    coeffs = pywt.wavedec2(attacked_channel, 'haar', level=2)
    _, (cH2_atk, _, _), _ = coeffs
    cH2_atk_flat = cH2_atk.flatten()
    
    weights = np.ones(4096, dtype=np.float32)
    
    # THE SOTA UPGRADE: Silence the corrupted votes!
    if tamper_map is not None:
        tamper_map_64 = cv2.resize(tamper_map, (64, 64), interpolation=cv2.INTER_NEAREST)
        tamper_flat = tamper_map_64.flatten()
        weights[tamper_flat == 255] = 0.0 # Revoke voting rights for tampered pixels
    
    # QIM BLIND EXTRACTION MATH
    step = np.round(cH2_atk_flat / delta)
    extracted_bits = (step % 2).astype(np.float32)
    
    # Apply Circuit Breaker weights
    extracted_bits *= weights
            
    reshaped_bits = extracted_bits.reshape(16, latent_dim)
    reshaped_weights = weights.reshape(16, latent_dim)
    
    # Tally the votes
    sum_votes = np.sum(reshaped_bits, axis=0)
    sum_weights = np.sum(reshaped_weights, axis=0)
    
    # Prevent division by zero if an attack is massive
    sum_weights[sum_weights == 0] = 1.0 
    
    final_bits = np.round(sum_votes / sum_weights).astype(np.uint8)
        
    decrypted_bits = process_watermark(final_bits, chaotic_key)
    
    latent_tensor = torch.tensor(decrypted_bits, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        recovered_img_tensor = decoder_model(latent_tensor)
        
    recovered_np = recovered_img_tensor.squeeze().cpu().numpy() * 255.0
    
    # NEW: Return the raw decrypted bits so we can calculate BER
    return recovered_np.astype(np.uint8), decrypted_bits