import numpy as np
import cv2
import random

def attack_crop(image, crop_percentage=0.10):
    """
    Simulates malicious object removal or catastrophic data loss.
    crop_percentage: float from 0.0 to 1.0 representing the area of the image to crop.
    """
    attacked = image.copy()
    h, w = attacked.shape
    
    # Calculate dimensions of the crop box to match the percentage area
    crop_side_ratio = np.sqrt(crop_percentage)
    crop_h = int(h * crop_side_ratio)
    crop_w = int(w * crop_side_ratio)
    
    # Calculate center coordinates for the blackout
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    
    attacked[start_y:start_y+crop_h, start_x:start_x+crop_w] = 0
    return attacked

def attack_text_insertion(image):
    """Simulates adding forged text or a fake watermark over the image"""
    attacked = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(attacked, "FORGED", (30, 130), font, 1.5, (255), 3, cv2.LINE_AA)
    return attacked

def attack_face_swap_sim(image):
    """Simulates a GenAI Inpainting / Face Swap by replacing a region with heavily modified pixels"""
    attacked = image.copy()
    patch = attacked[70:180, 70:180]
    # Blurring and shifting pixel intensities to simulate a fake patch
    forged_patch = cv2.GaussianBlur(patch, (25, 25), 0)
    forged_patch = cv2.add(forged_patch, 30) 
    attacked[70:180, 70:180] = np.clip(forged_patch, 0, 255)
    return attacked

def attack_copy_move(image):
    """Simulates internal forgery (copying one area to hide another)"""
    attacked = image.copy()
    # Copy a 60x60 patch from the top-left to the center
    patch = attacked[10:70, 10:70]
    attacked[100:160, 100:160] = patch
    return attacked

def attack_median_filter(image, kernel_size=5):
    """Simulates an adversary trying to smooth out/erase the watermark"""
    return cv2.medianBlur(image, kernel_size)

def attack_jpeg_compression(image, quality=60):
    """Simulates social media uploading/compression (Global Attack)"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)

def attack_noise_gaussian(image, std=15):
    """Gaussian noise with variable standard deviation (density)"""
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    attacked = cv2.add(image.astype(np.float32), noise)
    return np.clip(attacked, 0, 255).astype(np.uint8)

def attack_noise_sp(image, prob=0.05):
    """Salt and pepper noise with variable probability (density)"""
    attacked = image.copy()
    thres = 1 - prob
    # Vectorized for extreme speed
    rdn = np.random.rand(*image.shape)
    attacked[rdn < prob] = 0
    attacked[rdn > thres] = 255
    return attacked

def attack_speckle_noise(image, intensity=0.1):
    """Speckle noise with variable intensity"""
    noise = np.random.randn(*image.shape)
    attacked = image.astype(np.float32) + image.astype(np.float32) * noise * intensity
    return np.clip(attacked, 0, 255).astype(np.uint8)