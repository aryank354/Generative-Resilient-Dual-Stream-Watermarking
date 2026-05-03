import numpy as np
import pywt
import hashlib

def embed_robust_watermark(image_channel, encrypted_bits, delta=16.0):
    """
    Upgraded to Blind QIM (Quantization Index Modulation)
    delta: Quantization step size (replaces alpha)
    """
    coeffs = pywt.wavedec2(image_channel, 'haar', level=2)
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    
    cH2_flat = cH2.flatten()
    
    # 16x Hyper-Redundancy
    repeated_bits = np.tile(encrypted_bits, 16)
    
    # QIM Embedding Math (Vectorized for speed)
    step = np.round(cH2_flat / delta)
    mismatch = (step % 2) != repeated_bits
    
    # If the step parity doesn't match our secret bit, shift it to the nearest correct bin
    shift_up = cH2_flat > (step * delta)
    step[mismatch & shift_up] += 1
    step[mismatch & ~shift_up] -= 1
    
    cH2_modified = (step * delta).reshape(cH2.shape)
    
    coeffs_modified = [cA2, (cH2_modified, cV2, cD2), (cH1, cV1, cD1)]
    
    # Notice we NO LONGER return the orig_cH2. It is now fully Blind!
    return pywt.waverec2(coeffs_modified, 'haar')

def embed_fragile_watermark(image):
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    watermarked_img = img_uint8.copy()
    
    for i in range(0, img_uint8.shape[0], 8):
        for j in range(0, img_uint8.shape[1], 8):
            block = img_uint8[i:i+8, j:j+8]
            msb_block = block & 0xFC 
            
            # UPGRADED: SHA-256 instead of MD5
            hash_val = int(hashlib.sha256(msb_block.tobytes()).hexdigest()[:2], 16)
            lsb_val = hash_val % 4 
            watermarked_img[i:i+8, j:j+8] = msb_block | lsb_val
            
    return watermarked_img