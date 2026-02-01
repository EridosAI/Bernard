# train_lora.py - Train LoRA adapters from workshop sessions
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

print("="*70)
print("WORKSHOP BERNARD - LoRA Training")
print("="*70)

# Configuration
LEARNING_RATE = 1e-5
BATCH_SIZE = 2
EPOCHS = 5
LORA_RANK = 16
LORA_ALPHA = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")

# Find all session data
print("\nScanning for training data...")
session_dirs = glob.glob("./data/sessions/session_*")

if not session_dirs:
    print("❌ No training sessions found!")
    print("Run bernard_integrated_v2.py first to collect training data.")
    exit()

print(f"✓ Found {len(session_dirs)} session(s)")

# Load all training examples
class WorkshopDataset(Dataset):
    def __init__(self, session_dirs):
        self.examples = []
        
        for session_dir in session_dirs:
            embedding_files = glob.glob(os.path.join(session_dir, "clip_*_embeddings.npz"))
            
            for emb_file in embedding_files:
                data = np.load(emb_file, allow_pickle=True)
                
                # Skip empty narrations
                narration = str(data['narration'])
                if narration.strip():
                    self.examples.append({
                        'vision': torch.from_numpy(data['vision']).float(),
                        'text': torch.from_numpy(data['text']).float(),
                        'narration': narration
                    })
        
        print(f"  Loaded {len(self.examples)} training examples")
        
        # Show some examples
        print("\n  Sample narrations:")
        for i, ex in enumerate(self.examples[:3]):
            print(f"    {i+1}. \"{ex['narration']}\"")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# Create dataset
dataset = WorkshopDataset(session_dirs)

if len(dataset) == 0:
    print("\n❌ No valid training examples found (all narrations were empty)")
    print("Run bernard_integrated_v2.py and speak during sessions.")
    exit()

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load base model
print("\nLoading V-JEPA 2 model...")
model = AutoModel.from_pretrained("./models/base/vjepa2")
model = model.to(device)

print("✓ Model loaded")

# Add projection layer to align vision (1024) and text (384) dimensions
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
print("✓ Alignment head created")

# Configure LoRA
print("\nConfiguring LoRA...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["query", "value"],  # Apply LoRA to attention query and value
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Optimizer
params = list(model.parameters()) + list(alignment_head.parameters())
optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)

print("\n" + "="*70)
print("TRAINING")
print("="*70)

# Training loop
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 50)
    
    model.train()
    alignment_head.train()
    
    epoch_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Get embeddings
        vision_emb = batch['vision'].to(device)
        text_emb = batch['text'].to(device)
        
        # Project to common space
        vision_proj, text_proj = alignment_head(vision_emb, text_emb)
        
        # Normalize
        vision_norm = F.normalize(vision_proj, dim=-1)
        text_norm = F.normalize(text_proj, dim=-1)
        
        # Contrastive loss (align vision with corresponding text)
        # Positive pairs: same index in batch
        # Negative pairs: different indices
        
        # Calculate similarity between vision and text embeddings
        # vision_norm: [batch_size, 512]
        # text_norm: [batch_size, 512]
        
        # Squeeze to remove extra dimensions
        vision_norm = vision_norm.squeeze(1) if vision_norm.dim() > 2 else vision_norm
        text_norm = text_norm.squeeze(1) if text_norm.dim() > 2 else text_norm
        
        similarity = torch.matmul(vision_norm, text_norm.T)  # [batch_size, batch_size]
        
        # Simple contrastive loss - maximize diagonal similarity
        # Diagonal elements should be 1 (matching pairs)
        diagonal = torch.diagonal(similarity)
        loss = F.mse_loss(diagonal, torch.ones_like(diagonal))
        
        # Add penalty for high off-diagonal values
        if similarity.size(0) > 1:
            mask = ~torch.eye(similarity.size(0), dtype=bool, device=device)
            off_diagonal = similarity[mask]
            loss += 0.1 * torch.mean(off_diagonal ** 2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / num_batches
    print(f"\n  Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

# Save the trained model
output_dir = f"./models/adapters/workshop_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

print(f"\nSaving model to {output_dir}...")
model.save_pretrained(output_dir)
torch.save(alignment_head.state_dict(), os.path.join(output_dir, "alignment_head.pt"))

print("✓ Model saved!")

print("\n" + "="*70)
print("SUCCESS!")
print("="*70)
print(f"\nYour workshop Bernard has learned from {len(dataset)} examples!")
print(f"\nTrained LoRA adapters saved to:")
print(f"  {output_dir}")
print("\nNext step: Create inference script to use the trained model")