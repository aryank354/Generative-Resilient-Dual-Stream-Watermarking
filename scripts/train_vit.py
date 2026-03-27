import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import your model and the Binarize function
from gr_dsw.models.vit_autoencoder import WatermarkViTAutoEncoder, BinarizeSTE

def load_training_data(raw_images_dir):
    images = []
    valid_exts = ('.png', '.tiff', '.tif', '.jpg', '.jpeg')
    for f in os.listdir(raw_images_dir):
        if f.lower().endswith(valid_exts):
            img_path = os.path.join(raw_images_dir, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (256, 256))
                images.append(img / 255.0)
                
    if not images:
        raise ValueError(f"No valid images found in {raw_images_dir}!")
        
    data = torch.tensor(np.array(images), dtype=torch.float32).unsqueeze(1)
    return data

def simulate_latent_attacks(latent_batch):
    """
    Simulates physical image attacks by corrupting the 256-bit latent vector.
    This forces the ViT Decoder to learn Semantic Hallucination.
    """
    attacked_latent = latent_batch.clone()
    batch_size = attacked_latent.size(0)
    
    for i in range(batch_size):
        # 33% chance for Clean, 33% for Crop Simulation, 33% for Noise Simulation
        attack_choice = np.random.choice(['clean', 'drop', 'flip'])
        
        if attack_choice == 'drop':
            # Simulates Localized Forgery / Cropping (Up to 40% of bits lost)
            drop_mask = (torch.rand(attacked_latent[i].shape, device=latent_batch.device) > 0.4).float()
            attacked_latent[i] *= drop_mask
            
        elif attack_choice == 'flip':
            # Simulates Global Degradation / JPEG / Gaussian Noise (Up to 15% bits flipped)
            flip_mask = (torch.rand(attacked_latent[i].shape, device=latent_batch.device) < 0.15).float()
            # Flips 1s to 0s, and 0s to 1s
            attacked_latent[i] = torch.abs(attacked_latent[i] - flip_mask) 
            
    return attacked_latent

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_images_dir = os.path.join(base_dir, "RawImages")
    model_save_path = os.path.join(base_dir, "gr_dsw", "models", "pretrained_vit.pth")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print(f"[*] Loading training data from {raw_images_dir}...")
    train_data = load_training_data(raw_images_dir).to(device)
    print(f"[*] Loaded {train_data.shape[0]} images for training.")

    model = WatermarkViTAutoEncoder(latent_dim=256).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("[*] Starting Adversarial Training (1000 epochs)...")
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward pass through Encoder
        latent = model.encoder(train_data)
        binary_latent = BinarizeSTE.apply(latent)
        
        # 2. INJECT ATTACKS: Corrupt the latent vector
        attacked_latent = simulate_latent_attacks(binary_latent)
        
        # 3. Forward pass through Decoder using CORRUPTED bits
        reconstructed = model.decoder(attacked_latent)
        
        # 4. Calculate loss against the ORIGINAL clean image
        loss = criterion(reconstructed, train_data)
        
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"\n[+] Model weights saved successfully to {model_save_path}")
    print("[+] The ViT is now trained to resist spatial and signal attacks!")

if __name__ == "__main__":
    train_model()