# test_zero_shot.py - Test zero-shot recognition with workshop vocabulary
"""
Tests whether the existing alignment head (trained on 3 clips) can do
any zero-shot recognition against a vocabulary of common workshop objects.
"""

import cv2
import torch
import torch.nn as nn
from transformers import AutoModel, AutoVideoProcessor
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# ============================================================================
# ALIGNMENT HEAD (must match training architecture)
# ============================================================================

class AlignmentHead(nn.Module):
    """Projects vision and text embeddings to shared space"""
    def __init__(self, vision_dim: int = 1024, text_dim: int = 384, shared_dim: int = 512):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)
    
    def forward_vision(self, x):
        return self.vision_proj(x)
    
    def forward_text(self, x):
        return self.text_proj(x)

# ============================================================================
# WORKSHOP VOCABULARY
# ============================================================================

WORKSHOP_VOCABULARY = [
    # Hand tools
    "hammer", "screwdriver", "pliers", "wrench", "socket wrench",
    "allen key", "hex key", "tape measure", "level", "square",
    "chisel", "file", "rasp", "saw", "hacksaw",
    
    # Power tools
    "drill", "impact driver", "angle grinder", "jigsaw", "circular saw",
    "sander", "router", "heat gun", "soldering iron",
    
    # Fasteners & hardware
    "screw", "nail", "bolt", "nut", "washer",
    "bracket", "hinge", "hook", "clamp", "vice",
    
    # Electrical
    "wire", "cable", "connector", "terminal", "fuse",
    "multimeter", "wire stripper", "crimping tool",
    
    # Common objects
    "pen", "pencil", "marker", "tape", "glue",
    "scissors", "knife", "box cutter", "flashlight",
    "battery", "phone", "remote control", "mug", "bottle",
    
    # Containers
    "box", "bucket", "bin", "tray", "drawer",
    "toolbox", "case", "bag",
    
    # Materials
    "wood", "metal", "plastic", "paper", "cardboard",
    
    # Safety
    "gloves", "safety glasses", "mask", "ear protection",
    
    # Electronics
    "circuit board", "resistor", "capacitor", "LED", "switch",
    "arduino", "raspberry pi", "sensor", "motor",
    
    # Your specific items (from training data)
    "blue pen", "white bucket", "lid", "medicine bottle",
]

# ============================================================================
# MAIN TEST
# ============================================================================

def main():
    print("=" * 70)
    print("ZERO-SHOT RECOGNITION TEST")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Paths - adjust these to match your setup
    model_path = "./models/base/vjepa2"
    adapter_path = "./models/adapters/workshop_lora_20260117_041817"
    alignment_head_path = os.path.join(adapter_path, "alignment_head.pt")
    
    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------
    print("\n1. Loading models...")
    
    print("   Loading V-JEPA 2...")
    base_model = AutoModel.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base_model, adapter_path).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(model_path)
    
    print("   Loading text encoder...")
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("   Loading alignment head...")
    alignment_head = AlignmentHead().to(device)
    
    if os.path.exists(alignment_head_path):
        alignment_head.load_state_dict(torch.load(alignment_head_path, map_location=device))
        print(f"   ✓ Loaded alignment head from {alignment_head_path}")
    else:
        print(f"   ⚠ No alignment head found at {alignment_head_path}")
        print(f"   Using untrained alignment head (results will be random)")
    
    alignment_head.eval()
    
    # -------------------------------------------------------------------------
    # Pre-encode vocabulary
    # -------------------------------------------------------------------------
    print(f"\n2. Pre-encoding vocabulary ({len(WORKSHOP_VOCABULARY)} items)...")
    
    vocab_embeddings = {}
    with torch.no_grad():
        for item in WORKSHOP_VOCABULARY:
            # Encode text
            text_emb = text_encoder.encode(item, convert_to_tensor=True).to(device)
            # Project to shared space
            projected = alignment_head.forward_text(text_emb.unsqueeze(0))
            vocab_embeddings[item] = projected.squeeze(0)
    
    print(f"   ✓ Encoded {len(vocab_embeddings)} vocabulary items")
    
    # -------------------------------------------------------------------------
    # Camera capture function
    # -------------------------------------------------------------------------
    def capture_and_encode():
        """Capture 64 frames and encode through V-JEPA + alignment head"""
        cap = cv2.VideoCapture(0)
        frames = []
        
        print("\n   Capturing 64 frames...")
        for i in range(64):
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        
        if len(frames) < 64:
            print(f"   ⚠ Only got {len(frames)} frames")
            return None
        
        frames = np.array(frames)
        
        # Process through V-JEPA
        inputs = processor(videos=list(frames), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model(**inputs)
            vision_emb = output.last_hidden_state.mean(dim=1)  # [1, 1024]
            # Project to shared space
            projected = alignment_head.forward_vision(vision_emb)
        
        return projected.squeeze(0)
    
    # -------------------------------------------------------------------------
    # Recognition function
    # -------------------------------------------------------------------------
    def recognize(vision_embedding, top_k: int = 10):
        """Compare vision embedding against vocabulary"""
        scores = {}
        
        for item, text_emb in vocab_embeddings.items():
            sim = torch.cosine_similarity(
                vision_embedding.unsqueeze(0),
                text_emb.unsqueeze(0)
            ).item()
            scores[item] = sim
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]
    
    # -------------------------------------------------------------------------
    # Interactive test loop
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("INTERACTIVE TEST")
    print("Hold up an object and press ENTER to capture")
    print("Type 'q' to quit")
    print("=" * 70)
    
    while True:
        user_input = input("\nPress ENTER to capture (or 'q' to quit): ").strip().lower()
        
        if user_input == 'q':
            break
        
        # Capture and encode
        vision_emb = capture_and_encode()
        
        if vision_emb is None:
            continue
        
        # Get top matches
        matches = recognize(vision_emb)
        
        print("\n   Top 10 matches:")
        print("   " + "-" * 40)
        for i, (item, score) in enumerate(matches, 1):
            bar = "█" * int(score * 20) if score > 0 else ""
            print(f"   {i:2}. {item:20} {score:+.3f} {bar}")
        
        # Analysis
        best_item, best_score = matches[0]
        second_item, second_score = matches[1]
        margin = best_score - second_score
        
        print("\n   Analysis:")
        print(f"   Best match: '{best_item}' with score {best_score:.3f}")
        print(f"   Margin over second: {margin:.3f}")
        
        if best_score > 0.5 and margin > 0.1:
            print(f"   → HIGH confidence: This looks like a {best_item}")
        elif best_score > 0.3 and margin > 0.05:
            print(f"   → MEDIUM confidence: Might be a {best_item}")
        else:
            print(f"   → LOW confidence: Can't reliably identify")
    
    print("\nTest complete.")

if __name__ == "__main__":
    main()