# test_pretrained.py - Test what V-JEPA already knows
import cv2
import torch
import numpy as np
from transformers import AutoModel, AutoVideoProcessor

# Load BASE model (no LoRA)
print("Loading base V-JEPA 2 (no training)...")
model = AutoModel.from_pretrained("./models/base/vjepa2")
processor = AutoVideoProcessor.from_pretrained("./models/base/vjepa2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print("\nCapture video of different objects...")
print("We'll see how similar the base model thinks they are")
print("(before any of your training)\n")

cap = cv2.VideoCapture(0)

objects = []

for i in range(3):
    input(f"\nShow object #{i+1}, press ENTER to capture...")
    
    frames = []
    for _ in range(64):
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            cv2.imshow('Camera', frame)
            cv2.waitKey(1)
    
    frames_array = np.array(frames)
    inputs = processor(videos=list(frames_array), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model(**inputs)
        embedding = output.last_hidden_state.mean(dim=1)
    
    name = input("What was that object? ")
    objects.append({'name': name, 'embedding': embedding})
    print(f"✓ Captured '{name}'")

cap.release()
cv2.destroyAllWindows()

# Compare similarities
print("\n" + "="*70)
print("PRE-TRAINED SIMILARITIES (before your training)")
print("="*70)

for i in range(len(objects)):
    for j in range(i+1, len(objects)):
        emb1 = objects[i]['embedding']
        emb2 = objects[j]['embedding']
        
        sim = torch.cosine_similarity(emb1, emb2).item()
        
        print(f"\n'{objects[i]['name']}' vs '{objects[j]['name']}':")
        print(f"  Similarity: {sim:.3f}")
        
        if sim > 0.8:
            print("  → Very similar (model thinks they're related)")
        elif sim > 0.6:
            print("  → Somewhat similar")
        else:
            print("  → Different")

print("\n" + "="*70)
print("This shows what V-JEPA knows WITHOUT your training!")
print("If objects are already similar, you need less training.")
print("If objects are already different, generalization is good.")