# download_model.py
from transformers import AutoModel, AutoVideoProcessor
import os

print("Downloading V-JEPA 2 model...")
print("This is ~1-2GB and will take 5-10 minutes.\n")

# Create directories
os.makedirs("./models/base", exist_ok=True)

try:
    # Correct model name
    model_name = "facebook/vjepa2-vitl-fpc64-256"
    
    print(f"Downloading model: {model_name}")
    print("Downloading model weights...")
    model = AutoModel.from_pretrained(model_name)
    
    print("Downloading processor...")
    processor = AutoVideoProcessor.from_pretrained(model_name)
    
    # Save locally
    print("\nSaving to local directory...")
    model.save_pretrained("./models/base/vjepa2")
    processor.save_pretrained("./models/base/vjepa2")
    
    print("\n✓ V-JEPA 2 model downloaded and saved successfully!")
    print(f"Location: {os.path.abspath('./models/base/vjepa2')}")
    print(f"\nModel info:")
    print(f"  - Name: {model_name}")
    print(f"  - Type: Vision Transformer Large (ViT-L)")
    print(f"  - Frames per clip: 64")
    print(f"  - Resolution: 256x256")
    
except Exception as e:
    print(f"\n⚠ Error downloading model: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Make sure transformers is up to date: pip install -U transformers")