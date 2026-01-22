# test_focused_capture.py - Capture with object focus
import cv2
import torch
import numpy as np
from transformers import AutoModel, AutoVideoProcessor

def focus_on_center(frame, zoom_factor=2.0):
    """Zoom into center of frame"""
    h, w = frame.shape[:2]
    
    # Calculate crop dimensions
    crop_h = int(h / zoom_factor)
    crop_w = int(w / zoom_factor)
    
    # Center crop
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    cropped = frame[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    # Resize back to original size
    return cv2.resize(cropped, (w, h))

print("Testing focused capture...")
print("Position object in CENTER of frame\n")

model = AutoModel.from_pretrained("./models/base/vjepa2")
processor = AutoVideoProcessor.from_pretrained("./models/base/vjepa2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

cap = cv2.VideoCapture(0)

objects = []

for i in range(3):
    input(f"\nPlace object #{i+1} in CENTER of frame, press ENTER...")
    
    frames = []
    for j in range(64):
        ret, frame = cap.read()
        if ret:
            # Apply zoom
            focused = focus_on_center(frame, zoom_factor=2.0)
            frame_rgb = cv2.cvtColor(focused, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # Show what it's capturing
            cv2.imshow('Focused View (2x zoom)', focused)
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

# Compare
print("\n" + "="*70)
print("FOCUSED CAPTURE SIMILARITIES")
print("="*70)

for i in range(len(objects)):
    for j in range(i+1, len(objects)):
        emb1 = objects[i]['embedding']
        emb2 = objects[j]['embedding']
        
        sim = torch.cosine_similarity(emb1, emb2).item()
        
        print(f"\n'{objects[i]['name']}' vs '{objects[j]['name']}':")
        print(f"  Similarity: {sim:.3f}")
        
        if sim > 0.8:
            print("  → Still very similar")
        elif sim > 0.6:
            print("  → Somewhat similar (BETTER!)")
        elif sim > 0.4:
            print("  → Different (GOOD!)")
        else:
            print("  → Very different (EXCELLENT!)")

print("\nIf similarities are LOWER with focused capture,")
print("that means objects are more distinguishable!")