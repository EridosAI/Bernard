# test_inference.py - Test the trained model
import torch
import torch.nn as nn
import cv2
import numpy as np
from transformers import AutoModel, AutoVideoProcessor
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# Set OpenCV to use optimized code
cv2.setNumThreads(4)

print("="*70)
print("WORKSHOP BERNARD - Inference Test")
print("="*70)

# Load trained model
ADAPTER_PATH = "./models/adapters/workshop_lora_20260117_041817"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

print("\nLoading models...")
print("  - Base V-JEPA 2...")
base_model = AutoModel.from_pretrained("./models/base/vjepa2")
processor = AutoVideoProcessor.from_pretrained("./models/base/vjepa2")

print("  - Loading trained LoRA adapters...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.to(device)
model.eval()

print("  - Text encoder...")
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

print("  - Alignment head...")
class AlignmentHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_proj = nn.Linear(1024, 512)
        self.text_proj = nn.Linear(384, 512)
        
    def forward(self, vision_emb, text_emb):
        vision_proj = self.vision_proj(vision_emb)
        text_proj = self.text_proj(text_emb)
        return vision_proj, text_proj

alignment_head = AlignmentHead().to(device)
alignment_head.load_state_dict(torch.load(f"{ADAPTER_PATH}/alignment_head.pt"))
alignment_head.eval()

print("✓ All models loaded!")

# Open camera
cap = cv2.VideoCapture(0)

print("\n" + "="*70)
print("INTERACTIVE TEST")
print("="*70)
print("\nInstructions:")
print("  1. Show something to the camera")
print("  2. Type what you think it is")
print("  3. See how confident the model is!")
print("\nTry showing: blue pen, white box, bucket, lid")
print("(These are what you taught it)\n")

input("Press ENTER to start...")

while True:
    try:
        print("\n" + "-"*70)
        print("Capturing video clip (camera may freeze briefly)...")
        
        # Capture 64 frames
        frames = []
        for i in range(64):
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                # Only update display occasionally to stay responsive
                if i % 16 == 0:
                    cv2.imshow('Workshop Camera', frame)
                    cv2.waitKey(1)
        
        print("✓ Captured video clip")
        
        # Process through V-JEPA
        frames_array = np.array(frames)
        video_inputs = processor(videos=list(frames_array), return_tensors="pt")
        video_inputs = {k: v.to(device) for k, v in video_inputs.items()}
        
        with torch.no_grad():
            vision_output = model(**video_inputs)
            vision_embedding = vision_output.last_hidden_state.mean(dim=1)
        
        # Get user's description
        description = input("\nWhat do you see? (or 'quit'): ").strip()
        
        if description.lower() in ['quit', 'exit', 'q']:
            break
        
        if not description:
            continue
        
        # Encode the description
        text_embedding = text_encoder.encode(description, convert_to_tensor=True)
        text_embedding = text_embedding.to(device)
        
        # Project through alignment head
        with torch.no_grad():
            vision_proj, text_proj = alignment_head(
                vision_embedding,
                text_embedding.unsqueeze(0)
            )
            
            # Squeeze extra dimensions
            vision_proj = vision_proj.squeeze(1) if vision_proj.dim() > 2 else vision_proj
            text_proj = text_proj.squeeze(1) if text_proj.dim() > 2 else text_proj
            
            # Calculate similarity
            vision_norm = vision_proj / vision_proj.norm()
            text_norm = text_proj / text_proj.norm()
            similarity = torch.cosine_similarity(vision_norm, text_norm).item()
        
        print(f"\n{'='*70}")
        print(f"  Your description: '{description}'")
        print(f"  Confidence: {similarity:.3f}")
        
        if similarity > 0.8:
            print(f"  ✓ HIGH confidence - Good match!")
        elif similarity > 0.6:
            print(f"  ~ MEDIUM confidence - Partial match")
        else:
            print(f"  ✗ LOW confidence - Doesn't match well")
        
        print(f"{'='*70}")
        
        # Try comparing with training examples
        print("\nComparing with training examples:")
        training_examples = [
            "blue pen",
            "white box",
            "bucket",
            "lid"
        ]
        
        for example in training_examples:
            example_emb = text_encoder.encode(example, convert_to_tensor=True).to(device)
            with torch.no_grad():
                _, example_proj = alignment_head(
                    vision_embedding,
                    example_emb.unsqueeze(0)
                )
                example_proj = example_proj.squeeze(1) if example_proj.dim() > 2 else example_proj
                example_norm = example_proj / example_proj.norm()
                sim = torch.cosine_similarity(vision_norm, example_norm).item()
            
            print(f"  '{example}': {sim:.3f}")
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
        break
    except Exception as e:
        print(f"\nError: {e}")
        continue

cap.release()
cv2.destroyAllWindows()
print("\nTest complete!")