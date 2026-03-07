import numpy as np
import pywt
import hashlib

def embed_robust_watermark(image_channel, encrypted_bits, alpha=25.0):
    coeffs = pywt.wavedec2(image_channel, 'haar', level=2)
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    
    cH2_flat = cH2.flatten()
    orig_cH2 = cH2_flat.copy()
    
    # 16x Hyper-Redundancy
    repeated_bits = np.tile(encrypted_bits, 16)
    
    for i in range(len(repeated_bits)):
        if repeated_bits[i] == 1:
            cH2_flat[i] += alpha
        else:
            cH2_flat[i] -= alpha
            
    cH2_modified = cH2_flat.reshape(cH2.shape)
    coeffs_modified = [cA2, (cH2_modified, cV2, cD2), (cH1, cV1, cD1)]
    return pywt.waverec2(coeffs_modified, 'haar'), orig_cH2

def embed_fragile_watermark(image):
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    watermarked_img = img_uint8.copy()
    
    for i in range(0, img_uint8.shape[0], 8):
        for j in range(0, img_uint8.shape[1], 8):
            block = img_uint8[i:i+8, j:j+8]
            msb_block = block & 0xFC 
            hash_val = int(hashlib.md5(msb_block.tobytes()).hexdigest()[:2], 16)
            lsb_val = hash_val % 4 
            watermarked_img[i:i+8, j:j+8] = msb_block | lsb_val
            
    return watermarked_img