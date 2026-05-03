import os
import zipfile

def download_medical_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "TrainingDataset")
    os.makedirs(target_dir, exist_ok=True)
    
    print("[*] Authenticating with Kaggle...")
    # This downloads the dataset directly to your TrainingDataset folder
    os.system(f"kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p {target_dir}")
    
    zip_path = os.path.join(target_dir, "chest-xray-pneumonia.zip")
    
    if os.path.exists(zip_path):
        print("[*] Download complete. Extracting images (this may take a minute)...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("[+] Extraction complete!")
        
        # Clean up the zip file to save space
        os.remove(zip_path)
        print(f"[+] Medical dataset is ready in: {target_dir}")
    else:
        print("[!] Error: Could not find downloaded zip file. Check your kaggle.json setup.")

if __name__ == "__main__":
    download_medical_dataset()