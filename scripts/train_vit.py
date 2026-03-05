import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gr_dsw.models.vit_autoencoder import WatermarkViTAutoEncoder

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
    data = torch.tensor(np.array(images), dtype=torch.float32).unsqueeze(1)
    return data

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_images_dir = os.path.join(base_dir, "RawImages")
    model_save_path = os.path.join(base_dir, "gr_dsw", "models", "pretrained_vit.pth")
    
    train_data = load_training_data(raw_images_dir).to(device)

    model = WatermarkViTAutoEncoder(latent_dim=256).to(device) # Initialize with 256
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("[*] Starting training (1000 epochs)...")
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, reconstructed = model(train_data)
        loss = criterion(reconstructed, train_data)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"[*] Model weights saved successfully to {model_save_path}")

if __name__ == "__main__":
    train_model()