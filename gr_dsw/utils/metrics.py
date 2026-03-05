import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def evaluate_quality(original, processed):
    p_val = psnr(original, processed, data_range=255)
    s_val = ssim(original, processed, data_range=255)
    return p_val, s_val