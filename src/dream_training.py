# dream_training.py - Overnight LoRA Training ("Dreaming")
"""
Processes training data collected during the day and trains LoRA adapters.
Run this overnight or when the system is idle.

Usage:
    python dream_training.py                    # Train on all available data
    python dream_training.py --session latest   # Train on latest session only
    python dream_training.py --epochs 20        # Custom epochs
"""

import os
import json
import argparse
import glob
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoVideoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================

class TrainingConfig:
    # Model paths
    base_model_path = "./models/base/vjepa2"
    existing_adapter_path = "./models/adapters/workshop_lora_20260117_041817"
    output_dir = "./models/adapters"
    
    # Training data
    training_data_dir = "./data/training"
    
    # Training params
    batch_size = 2
    num_epochs = 10
    learning_rate = 1e-4
    weight_decay = 0.01
    
    # LoRA params
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    target_modules = ["query", "value"]
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# DATASET
# ============================================================================

class WorkshopObjectDataset(Dataset):
    """Dataset of cropped object images with labels"""
    
    def __init__(self, data_dir: str, processor, frame_type: str = "confirmed"):
        """
        Args:
            data_dir: Path to training data directory
            processor: V-JEPA processor
            frame_type: "confirmed", "uncertain", or "all"
        """
        self.processor = processor
        self.frame_type = frame_type
        self.samples = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        # Find all session directories
        sessions = glob.glob(os.path.join(data_dir, "session_*"))
        
        for session_path in sessions:
            manifest_path = os.path.join(session_path, "manifest.json")
            if not os.path.exists(manifest_path):
                continue
            
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            for obj_name, obj_info in manifest.get('objects', {}).items():
                # Create label index
                if obj_name not in self.label_to_idx:
                    idx = len(self.label_to_idx)
                    self.label_to_idx[obj_name] = idx
                    self.idx_to_label[idx] = obj_name
                
                # Determine which directories to scan
                obj_base = os.path.join(session_path, obj_name.replace(' ', '_'))
                
                dirs_to_scan = []
                if frame_type in ["confirmed", "all"]:
                    confirmed_dir = os.path.join(obj_base, "confirmed")
                    if os.path.exists(confirmed_dir):
                        dirs_to_scan.append(confirmed_dir)
                    # Also check old format (no subdirectory)
                    elif os.path.exists(obj_base) and not os.path.isdir(os.path.join(obj_base, "confirmed")):
                        # Old format - frames directly in object folder
                        old_frames = glob.glob(os.path.join(obj_base, "frame_*.jpg"))
                        if old_frames:
                            dirs_to_scan.append(obj_base)
                
                if frame_type in ["uncertain", "all"]:
                    uncertain_dir = os.path.join(obj_base, "uncertain")
                    if os.path.exists(uncertain_dir):
                        dirs_to_scan.append(uncertain_dir)
                
                # Find all frames
                for scan_dir in dirs_to_scan:
                    frame_paths = sorted(glob.glob(os.path.join(scan_dir, "frame_*.jpg")))
                    is_uncertain = "uncertain" in scan_dir
                    
                    for frame_path in frame_paths:
                        self.samples.append({
                            'path': frame_path,
                            'label': obj_name,
                            'label_idx': self.label_to_idx[obj_name],
                            'category': obj_info.get('category', 'unknown'),
                            'uncertain': is_uncertain
                        })
        
        confirmed_count = sum(1 for s in self.samples if not s.get('uncertain', False))
        uncertain_count = sum(1 for s in self.samples if s.get('uncertain', False))
        print(f"Loaded {len(self.samples)} samples across {len(self.label_to_idx)} objects")
        print(f"  Confirmed: {confirmed_count}, Uncertain: {uncertain_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = cv2.imread(sample['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to V-JEPA expected size
        img = cv2.resize(img, (256, 256))
        
        # Create a fake "video" by repeating the frame
        # V-JEPA expects video input
        frames = np.stack([img] * 16, axis=0)  # 16 frames
        
        # Process
        inputs = self.processor(videos=list(frames), return_tensors="pt")
        
        # Use the pixel_key if set, otherwise try to find it
        pixel_key = getattr(self, 'pixel_key', None)
        
        if pixel_key is None:
            for key in ['pixel_values', 'pixel_values_videos', 'input_values']:
                if key in inputs:
                    pixel_key = key
                    break
        
        if pixel_key is None:
            # Use first 4D+ tensor found
            for key, val in inputs.items():
                if isinstance(val, torch.Tensor) and val.dim() >= 4:
                    pixel_key = key
                    break
        
        if pixel_key is None:
            raise KeyError(f"Could not find pixel values. Keys: {list(inputs.keys())}")
        
        return {
            'pixel_values': inputs[pixel_key].squeeze(0),
            'label_idx': sample['label_idx'],
            'label': sample['label'],
            'path': sample['path'],
            'uncertain': sample.get('uncertain', False)
        }

# ============================================================================
# CONTRASTIVE LOSS
# ============================================================================

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning discriminative embeddings.
    Same object views should be similar, different objects should be different.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size] - integer labels
        """
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(mask.size(0), device=mask.device)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        
        # For each sample, compute log probability of positive pairs
        # vs all pairs
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Average over positive pairs
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        loss = -mean_log_prob.mean()
        
        return loss

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(config: TrainingConfig, session_filter: str = None):
    """Main training function"""
    
    print("=" * 70)
    print("DREAM TRAINING - Overnight LoRA Update")
    print("=" * 70)
    print(f"\nDevice: {config.device}")
    print(f"Training data: {config.training_data_dir}")
    
    # Check for training data
    if not os.path.exists(config.training_data_dir):
        print(f"\n❌ No training data found at {config.training_data_dir}")
        print("   Run the main system and teach it some objects first!")
        return
    
    sessions = glob.glob(os.path.join(config.training_data_dir, "session_*"))
    if not sessions:
        print(f"\n❌ No training sessions found")
        return
    
    print(f"Found {len(sessions)} training sessions")
    
    # Load processor
    print("\nLoading V-JEPA processor...")
    processor = AutoVideoProcessor.from_pretrained(config.base_model_path)
    
    # Load dataset
    print("Loading training data...")
    dataset = WorkshopObjectDataset(config.training_data_dir, processor)
    
    if len(dataset) < 2:
        print(f"\n❌ Not enough training samples ({len(dataset)})")
        print("   Need at least 2 samples for contrastive learning")
        return
    
    # Test processor output format
    print("Testing processor output format...")
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    test_frames = np.stack([test_img] * 16, axis=0)
    test_inputs = processor(videos=list(test_frames), return_tensors="pt")
    print(f"  Processor returns keys: {list(test_inputs.keys())}")
    
    # Find the pixel values key
    pixel_key = None
    for k, v in test_inputs.items():
        if hasattr(v, 'shape'):
            print(f"    {k}: {v.shape}")
            if pixel_key is None and v.dim() >= 4:
                pixel_key = k
    
    if pixel_key is None:
        print("❌ Could not determine pixel values key from processor")
        return
    
    print(f"  Using key: '{pixel_key}'")
    
    # Store for dataset to use
    dataset.pixel_key = pixel_key
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        drop_last=True  # Need even batches for contrastive loss
    )
    
    # Load model
    print("\nLoading V-JEPA model...")
    base_model = AutoModel.from_pretrained(config.base_model_path)
    
    # Check if we have existing adapters to continue from
    if os.path.exists(config.existing_adapter_path):
        print(f"Loading existing adapters from {config.existing_adapter_path}")
        model = PeftModel.from_pretrained(base_model, config.existing_adapter_path)
        # Unfreeze for continued training
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
    else:
        print("Creating new LoRA adapters...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)
    
    model = model.to(config.device)
    model.train()
    
    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Loss
    criterion = ContrastiveLoss(temperature=0.07)
    
    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    print("-" * 50)
    
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(config.device)
            labels = batch['label_idx'].to(config.device)
            
            # Forward pass - V-JEPA expects the input under the pixel_key name
            outputs = model(**{pixel_key: pixel_values})
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Pool over sequence
            
            # Compute loss
            loss = criterion(embeddings, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config.output_dir, f"workshop_lora_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    model.save_pretrained(output_path)
    
    print("-" * 50)
    print(f"\n✓ Training complete!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Model saved to: {output_path}")
    
    # ========================================================================
    # PHASE 2: MEMORY CONSOLIDATION - Review uncertain frames
    # ========================================================================
    
    print("\n" + "=" * 50)
    print("PHASE 2: Memory Consolidation")
    print("=" * 50)
    
    # Load uncertain frames
    uncertain_dataset = WorkshopObjectDataset(config.training_data_dir, processor, frame_type="uncertain")
    uncertain_dataset.pixel_key = pixel_key
    
    if len(uncertain_dataset) == 0:
        print("No uncertain frames to consolidate")
    else:
        print(f"\nReviewing {len(uncertain_dataset)} uncertain frames...")
        
        # Get embeddings for all confirmed frames (for comparison)
        confirmed_dataset = WorkshopObjectDataset(config.training_data_dir, processor, frame_type="confirmed")
        confirmed_dataset.pixel_key = pixel_key
        
        # Build embedding bank for each object from confirmed frames
        model.eval()
        embedding_bank = {}  # {label: [embeddings]}
        
        print("Building embedding bank from confirmed frames...")
        with torch.no_grad():
            for i in tqdm(range(len(confirmed_dataset)), desc="Embedding confirmed"):
                sample = confirmed_dataset[i]
                pixel_values = sample['pixel_values'].unsqueeze(0).to(config.device)
                label = sample['label']
                
                outputs = model(**{pixel_key: pixel_values})
                emb = outputs.last_hidden_state.mean(dim=1).cpu()
                
                if label not in embedding_bank:
                    embedding_bank[label] = []
                embedding_bank[label].append(emb)
        
        # Review uncertain frames
        promoted = 0
        rejected = 0
        promotion_threshold = 0.50  # Must be this similar to confirmed frames
        
        print("\nReviewing uncertain frames...")
        frames_to_promote = []
        
        with torch.no_grad():
            for i in tqdm(range(len(uncertain_dataset)), desc="Reviewing uncertain"):
                sample = uncertain_dataset[i]
                pixel_values = sample['pixel_values'].unsqueeze(0).to(config.device)
                label = sample['label']
                frame_path = sample['path']
                
                if label not in embedding_bank:
                    # No confirmed frames for this object - can't verify
                    rejected += 1
                    continue
                
                # Get embedding for uncertain frame
                outputs = model(**{pixel_key: pixel_values})
                uncertain_emb = outputs.last_hidden_state.mean(dim=1).cpu()
                
                # Compare to confirmed embeddings
                max_sim = 0.0
                for confirmed_emb in embedding_bank[label]:
                    sim = torch.cosine_similarity(
                        uncertain_emb.flatten().unsqueeze(0),
                        confirmed_emb.flatten().unsqueeze(0)
                    ).item()
                    max_sim = max(max_sim, sim)
                
                if max_sim >= promotion_threshold:
                    promoted += 1
                    frames_to_promote.append({
                        'path': frame_path,
                        'label': label,
                        'similarity': max_sim
                    })
                else:
                    rejected += 1
        
        print(f"\n📊 Consolidation Results:")
        print(f"   Promoted: {promoted} frames (similarity >= {promotion_threshold})")
        print(f"   Rejected: {rejected} frames")
        
        # Move promoted frames to confirmed directory
        if frames_to_promote:
            print(f"\n📁 Moving {len(frames_to_promote)} frames to confirmed...")
            
            for frame_info in frames_to_promote:
                src_path = frame_info['path']
                
                # Build destination path (replace 'uncertain' with 'confirmed')
                dst_path = src_path.replace('/uncertain/', '/confirmed/').replace('\\uncertain\\', '\\confirmed\\')
                
                # Ensure destination directory exists
                dst_dir = os.path.dirname(dst_path)
                os.makedirs(dst_dir, exist_ok=True)
                
                # Move the file
                try:
                    shutil.move(src_path, dst_path)
                except Exception as e:
                    print(f"   Warning: Could not move {src_path}: {e}")
            
            print(f"   ✓ Promoted frames will be used in next training session")
    
    # ========================================================================
    # PHASE 3: ASSOCIATIVE MEMORY - Train Inward JEPA
    # ========================================================================
    
    print("\n" + "=" * 50)
    print("PHASE 3: Associative Memory Training")
    print("=" * 50)
    
    try:
        from .episode_memory import EpisodeMemory
        from .inward_jepa import InwardJEPA, InwardJEPATrainer, train_inward_jepa
    except ImportError:
        from episode_memory import EpisodeMemory
        from inward_jepa import InwardJEPA, InwardJEPATrainer, train_inward_jepa
    
    # Load episode memory
    episode_memory_path = "./data/episode_memory.json"
    
    if os.path.exists(episode_memory_path):
        episode_memory = EpisodeMemory(save_path=episode_memory_path)
        
        # Check if we have enough data
        episodes_with_emb = sum(1 for ep in episode_memory.episodes if ep.embedding is not None)
        unique_objects = len(episode_memory.spatial_priors)
        
        print(f"Episodes with embeddings: {episodes_with_emb}")
        print(f"Unique objects: {unique_objects}")
        
        # Need sufficient diversity for contrastive learning
        if episodes_with_emb >= 50 and unique_objects >= 10:
            # Determine embedding dimension from first episode
            emb_dim = None
            for ep in episode_memory.episodes:
                if ep.embedding is not None:
                    emb_dim = ep.embedding.shape[0]
                    break
            
            if emb_dim:
                print(f"Embedding dimension: {emb_dim}")
                
                # Initialize or load inward JEPA
                inward_jepa_path = "./models/inward_jepa_weights.pt"
                inward_model = InwardJEPA(embedding_dim=emb_dim)
                
                if os.path.exists(inward_jepa_path):
                    print(f"Loading existing inward JEPA weights...")
                    inward_model.load_state_dict(torch.load(inward_jepa_path))
                
                inward_model = inward_model.to(config.device)
                inward_trainer = InwardJEPATrainer(inward_model)
                
                # Train
                print(f"\nTraining associative memory...")
                train_inward_jepa(
                    episode_memory, 
                    inward_model, 
                    inward_trainer, 
                    epochs=20,
                    batch_size=16
                )
                
                # Save weights
                os.makedirs(os.path.dirname(inward_jepa_path), exist_ok=True)
                torch.save(inward_model.state_dict(), inward_jepa_path)
                print(f"✓ Inward JEPA saved to {inward_jepa_path}")
            else:
                print("⚠ Could not determine embedding dimension")
        else:
            print(f"⚠ Not enough data for associative training")
            print(f"  Need: 50+ episodes with embeddings, 10+ unique objects")
            print(f"  Have: {episodes_with_emb} episodes, {unique_objects} objects")
    else:
        print(f"⚠ No episode memory found at {episode_memory_path}")
    
    print("\n" + "=" * 50)
    print(f"To use the new model, update the adapter path in arnold_integrated_v2.py:")
    print(f'  adapter_path = "{output_path}"')

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overnight LoRA training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--session", type=str, default=None, help="Specific session to train on")
    
    args = parser.parse_args()
    
    config = TrainingConfig()
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    
    train(config, session_filter=args.session)
