import numpy as np
import cv2
import random

def attack_crop(image):
    """Simulates malicious object removal (Data Loss)"""
    attacked = image.copy()
    attacked[90:160, 90:160] = 0 # Blackout a central block
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

def attack_jpeg_compression(image, quality=60):
    """Simulates social media uploading/compression (Global Attack)"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)

def attack_noise_gaussian(image, mean=0, std=15):
    """Simulates sensor/transmission noise (Global Attack)"""
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    attacked = cv2.add(image.astype(np.float32), noise)
    return np.clip(attacked, 0, 255).astype(np.uint8)

def attack_noise_sp(image, prob=0.05):
    """Simulates Salt & Pepper impulse noise (Global Attack)"""
    attacked = image.copy()
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                attacked[i][j] = 0
            elif rdn > thres:
                attacked[i][j] = 255
    return attacked