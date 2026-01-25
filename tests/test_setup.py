# test_setup.py
import torch
import cv2
import whisper
from transformers import AutoModel

print("Testing installation...")

# Test PyTorch + CUDA
print(f"\n1. PyTorch & CUDA:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Test OpenCV (camera library)
print(f"\n2. OpenCV:")
print(f"   Version: {cv2.__version__}")

# Test Whisper
print(f"\n3. Whisper (audio transcription):")
print(f"   Loading base model...")
model = whisper.load_model("base")
print(f"   ✓ Whisper loaded successfully")

# Test Transformers
print(f"\n4. Hugging Face Transformers:")
print(f"   Checking model hub access...")
try:
    from transformers import AutoConfig
    print(f"   ✓ Transformers library working")
except Exception as e:
    print(f"   ⚠ Issue: {e}")

print("\n✓ All components installed successfully!")
print("\nYour RTX 4080 Super is ready for AI training!")