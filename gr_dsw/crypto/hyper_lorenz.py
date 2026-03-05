import numpy as np
from scipy.integrate import solve_ivp

def _hyper_lorenz_derivatives(t, state, a=10, b=8/3, c=28, r=-1):
    x, y, z, w = state
    return [
        a * (y - x) + w,
        c * x - y - x * z,
        x * y - b * z,
        -x * z + r * w
    ]

def generate_chaotic_key(length, initial_state=[1.0, 2.0, 3.0, 4.0]):
    t_span = (0, 100)
    t_eval = np.linspace(0, 100, length + 1000)
    sol = solve_ivp(_hyper_lorenz_derivatives, t_span, initial_state, t_eval=t_eval)
    
    x_val = sol.y[0][1000:]
    threshold = np.median(x_val)
    return (x_val > threshold).astype(np.uint8)

def process_watermark(watermark_bits, chaotic_key):
    """Encrypts or Decrypts via XOR"""
    return np.bitwise_xor(watermark_bits.astype(np.uint8), chaotic_key)