# test_vjepa_camera.py
import cv2
import torch
from transformers import AutoModel, AutoVideoProcessor
import numpy as np

print("Loading V-JEPA 2 model...")
model = AutoModel.from_pretrained("./models/base/vjepa2")
processor = AutoVideoProcessor.from_pretrained("./models/base/vjepa2")

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()  # Set to evaluation mode

print(f"✓ Model loaded on {device}")
print(f"  GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")

# Open camera
cap = cv2.VideoCapture(0)
print("\n✓ Camera opened")

print("\nCapturing 64 frames for V-JEPA processing...")
print("(Hold still for ~2 seconds)\n")

frames = []
for i in range(64):
    ret, frame = cap.read()
    if ret:
        # Convert BGR (OpenCV) to RGB (model expects)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
        # Show progress
        if i % 16 == 0:
            print(f"  Captured {i}/64 frames...")
    else:
        print(f"Failed to capture frame {i}")
        break

cap.release()

if len(frames) == 64:
    print(f"\n✓ Captured {len(frames)} frames")
    
    # Process frames through V-JEPA
    print("\nProcessing through V-JEPA 2...")
    
    # Prepare input (add batch dimension)
    frames_array = np.array(frames)
    inputs = processor(videos=list(frames_array), return_tensors="pt")
    
    # Move to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embeddings
    embeddings = outputs.last_hidden_state
    
    print("✓ Processing complete!")
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"  - Batch size: {embeddings.shape[0]}")
    print(f"  - Sequence length: {embeddings.shape[1]}")
    print(f"  - Embedding dimension: {embeddings.shape[2]}")
    
    # Show some statistics about the embedding
    embedding_mean = embeddings.mean().item()
    embedding_std = embeddings.std().item()
    
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embedding_mean:.4f}")
    print(f"  Std dev: {embedding_std:.4f}")
    
    print("\n" + "="*60)
    print("SUCCESS! V-JEPA 2 can process your camera feed!")
    print("="*60)
    print("\nThis embedding represents what V-JEPA 'understands'")
    print("about your 64-frame video clip in abstract feature space.")
    print("\nNext step: Add speech-to-text and start training!")

else:
    print(f"\n❌ Only captured {len(frames)} frames (needed 64)")

cv2.destroyAllWindows()