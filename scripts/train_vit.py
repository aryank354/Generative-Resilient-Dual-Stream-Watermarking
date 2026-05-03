import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import your model and the Binarize function
from gr_dsw.models.vit_autoencoder import WatermarkViTAutoEncoder, BinarizeSTE

class SecureImageDataset(Dataset):
    def __init__(self, raw_images_dir, image_size=(256, 256)):
        self.image_paths = []
        self.image_size = image_size
        valid_exts = ('.png', '.tiff', '.tif', '.jpg', '.jpeg')
        
        # Recursively find all images in the dataset folder
        for root, _, files in os.walk(raw_images_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    self.image_paths.append(os.path.join(root, f))
                    
        if not self.image_paths:
            raise ValueError(f"No valid images found in {raw_images_dir}!")
            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Handle corrupted images gracefully
        if img is None:
            img = np.zeros(self.image_size, dtype=np.uint8)
            
        img = cv2.resize(img, self.image_size)
        img_tensor = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)
        return img_tensor

def simulate_latent_attacks(latent_batch):
    """
    Forces the Decoder to learn Semantic Hallucination by simulating physical attacks on the bits.
    """
    attacked_latent = latent_batch.clone()
    batch_size = attacked_latent.size(0)
    
    for i in range(batch_size):
        attack_choice = np.random.choice(['clean', 'drop', 'flip'])
        
        if attack_choice == 'drop':
            drop_mask = (torch.rand(attacked_latent[i].shape, device=latent_batch.device) > 0.4).float()
            attacked_latent[i] *= drop_mask
            
        elif attack_choice == 'flip':
            flip_mask = (torch.rand(attacked_latent[i].shape, device=latent_batch.device) < 0.15).float()
            attacked_latent[i] = torch.abs(attacked_latent[i] - flip_mask) 
            
    return attacked_latent

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Update to point to the new dataset folder
    dataset_dir = os.path.join(base_dir, "TrainingDataset")
    model_save_path = os.path.join(base_dir, "gr_dsw", "models", "pretrained_vit.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print(f"[*] Initializing Dataset from {dataset_dir}...")
    dataset = SecureImageDataset(dataset_dir)
    print(f"[*] Found {len(dataset)} images for training.")
    
    # DataLoader handles batching automatically (adjust batch_size if you run out of GPU memory)
    batch_size = 32 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = WatermarkViTAutoEncoder(latent_dim=256).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200 # 200 epochs on 5,800 images is plenty
    print(f"[*] Starting Adversarial Training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Professional progress bar for each epoch
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            latent = model.encoder(data)
            binary_latent = BinarizeSTE.apply(latent)
            
            # Adversarial injection
            attacked_latent = simulate_latent_attacks(binary_latent)
            reconstructed = model.decoder(attacked_latent)
            
            # Loss and backprop
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{(running_loss / (batch_idx + 1)):.5f}"})
            
        # Save checkpoints every 50 epochs just in case
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), model_save_path.replace(".pth", f"_epoch{epoch+1}.pth"))

    # Save final model
    torch.save(model.state_dict(), model_save_path)
    print(f"\n[+] Final Model weights saved successfully to {model_save_path}")

if __name__ == "__main__":
    train_model()