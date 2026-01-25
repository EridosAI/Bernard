"""
Audio-Visual Contrastive Training

Trains a projection head to align BEATs audio embeddings with V-JEPA visual embeddings
using InfoNCE contrastive loss.

Architecture:
    BEATs (768-dim, frozen) -> Projection Head (trainable) -> V-JEPA space (1024-dim)

Training:
    - InfoNCE loss: paired audio-visual samples are positives
    - All other samples in batch are negatives
    - Only projection head is trained; both encoders stay frozen
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================================
# PROJECTION HEAD
# ============================================================================

class AudioVisualProjection(nn.Module):
    """
    Projects BEATs audio embeddings (768) to V-JEPA visual space (1024).

    Architecture:
        Linear(768, 512) -> GELU -> Dropout
        Linear(512, 512) -> GELU -> Dropout
        Linear(512, 1024)

    This is the ONLY trainable component in the audio-visual alignment pipeline.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, audio_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project audio embedding to visual space.

        Args:
            audio_embedding: [batch, 768] or [768]

        Returns:
            projected: [batch, 1024] or [1024]
        """
        return self.projection(audio_embedding)


# ============================================================================
# INFONCE LOSS
# ============================================================================

class AudioVisualInfoNCE(nn.Module):
    """
    InfoNCE contrastive loss for audio-visual alignment.

    Given a batch of (audio, visual) pairs:
    - Each audio has exactly one positive visual (its pair at same index)
    - All other visuals in the batch are negatives

    Loss is symmetric: audio->visual and visual->audio directions.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        audio_embeddings: torch.Tensor,
        visual_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute symmetric InfoNCE loss.

        Args:
            audio_embeddings: [batch, dim] - projected audio embeddings
            visual_embeddings: [batch, dim] - V-JEPA visual embeddings

        Returns:
            loss: scalar tensor
        """
        # Normalize embeddings
        audio_norm = F.normalize(audio_embeddings, dim=1)
        visual_norm = F.normalize(visual_embeddings, dim=1)

        # Compute similarity matrix [batch, batch]
        # sim[i,j] = cosine similarity between audio_i and visual_j
        sim_matrix = torch.matmul(audio_norm, visual_norm.T) / self.temperature

        # Labels: diagonal elements are positives (audio_i matches visual_i)
        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=sim_matrix.device)

        # Audio-to-visual loss: for each audio, find its visual
        loss_a2v = F.cross_entropy(sim_matrix, labels)

        # Visual-to-audio loss: for each visual, find its audio
        loss_v2a = F.cross_entropy(sim_matrix.T, labels)

        return (loss_a2v + loss_v2a) / 2


# ============================================================================
# TRAINING CONFIG
# ============================================================================

@dataclass
class AudioVisualTrainingConfig:
    """Configuration for audio-visual training"""

    # Data
    data_dir: str = "./data/vggsound"

    # Model paths
    vjepa_path: str = "./models/base/vjepa2"
    vjepa_adapter_path: str = "./models/adapters/workshop_lora_20260117_041817"
    beats_checkpoint: str = "./models/base/beats/BEATs_iter3_plus_AS2M.pt"

    # Architecture
    hidden_dim: int = 512
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    temperature: float = 0.07

    # Evaluation
    eval_every: int = 5  # epochs

    # Output
    output_dir: str = "./models/adapters/audio_visual_projection"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# TRAINER
# ============================================================================

class AudioVisualTrainer:
    """
    Trains projection head to align BEATs audio with V-JEPA visual embeddings.

    Both encoders stay FROZEN. Only the projection head is trained.
    """

    def __init__(self, config: AudioVisualTrainingConfig = None):
        self.config = config or AudioVisualTrainingConfig()
        self.device = self.config.device

        # Will be initialized in setup()
        self.beats_encoder = None
        self.vjepa_encoder = None
        self.projection = None
        self.criterion = None
        self.optimizer = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None

        # Training history
        self.history = {
            'config': asdict(self.config),
            'epochs': [],
            'best_epoch': 0,
            'best_top1_accuracy': 0.0
        }

    def setup(self):
        """Initialize models, datasets, and optimizers"""
        print("=" * 70)
        print("Audio-Visual Contrastive Training Setup")
        print("=" * 70)

        # Load encoders
        print("\nLoading encoders...")
        self._load_encoders()

        # Create projection head
        print("\nCreating projection head...")
        self.projection = AudioVisualProjection(
            input_dim=768,  # BEATs
            hidden_dim=self.config.hidden_dim,
            output_dim=1024,  # V-JEPA
            dropout=self.config.dropout
        ).to(self.device)

        params = sum(p.numel() for p in self.projection.parameters())
        print(f"  Projection head: {params:,} parameters")

        # Loss and optimizer
        self.criterion = AudioVisualInfoNCE(temperature=self.config.temperature)
        self.optimizer = torch.optim.AdamW(
            self.projection.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Load datasets
        print("\nLoading datasets...")
        self._load_datasets()

        print("\nSetup complete!")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Device: {self.device}")

    def _load_encoders(self):
        """Load BEATs and V-JEPA encoders"""
        try:
            from .beats_encoder import BEATsEncoder
        except ImportError:
            from beats_encoder import BEATsEncoder

        # BEATs
        self.beats_encoder = BEATsEncoder(
            checkpoint_path=self.config.beats_checkpoint,
            device=self.device
        )

        # V-JEPA - use existing VJEPAEncoder from project
        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, str(Path(__file__).parent))

        try:
            from jarvis_integrated_v2 import VJEPAEncoder
            self.vjepa_encoder = VJEPAEncoder(
                self.config.vjepa_path,
                self.config.vjepa_adapter_path,
                device=self.device
            )
        except ImportError:
            # Fallback: create minimal V-JEPA encoder
            print("  Warning: Could not import VJEPAEncoder, using minimal version")
            self.vjepa_encoder = self._create_minimal_vjepa_encoder()

    def _create_minimal_vjepa_encoder(self):
        """Create minimal V-JEPA encoder without full jarvis integration"""
        from transformers import AutoModel, AutoVideoProcessor
        from peft import PeftModel

        class MinimalVJEPAEncoder:
            def __init__(self, model_path, adapter_path, device):
                self.device = device
                base_model = AutoModel.from_pretrained(model_path)
                self.model = PeftModel.from_pretrained(base_model, adapter_path).to(device).eval()
                self.processor = AutoVideoProcessor.from_pretrained(model_path)

                for param in self.model.parameters():
                    param.requires_grad = False

            def encode_frames(self, frames):
                import numpy as np
                inputs = self.processor(videos=list(frames), return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    output = self.model(**inputs)
                    embedding = output.last_hidden_state.mean(dim=1)

                return embedding.squeeze(0)

        return MinimalVJEPAEncoder(
            self.config.vjepa_path,
            self.config.vjepa_adapter_path,
            self.device
        )

    def _load_datasets(self):
        """Load train and test datasets with precomputed embeddings"""
        from audio_visual_dataset import VGGSoundDataset

        self.train_dataset = VGGSoundDataset(
            data_dir=self.config.data_dir,
            beats_encoder=self.beats_encoder,
            vjepa_encoder=self.vjepa_encoder,
            precompute_embeddings=True,
            split="train"
        )

        self.test_dataset = VGGSoundDataset(
            data_dir=self.config.data_dir,
            beats_encoder=self.beats_encoder,
            vjepa_encoder=self.vjepa_encoder,
            precompute_embeddings=True,
            split="test"
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,  # Need full batches for InfoNCE
            num_workers=0
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )

    def train_epoch(self) -> float:
        """Train one epoch, return average loss"""
        self.projection.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            audio_emb = batch['audio_embedding'].to(self.device)
            visual_emb = batch['visual_embedding'].to(self.device)

            # Project audio to visual space
            audio_projected = self.projection(audio_emb)

            # Compute InfoNCE loss
            loss = self.criterion(audio_projected, visual_emb)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate_retrieval(self) -> Dict[str, float]:
        """
        Evaluate audio->visual retrieval accuracy on test set.

        For each audio embedding:
        1. Project to visual space
        2. Find nearest visual embedding
        3. Check if it's the correct pair

        Returns:
            {
                'top1_accuracy': float,
                'top5_accuracy': float,
                'mean_rank': float
            }
        """
        self.projection.eval()

        # Collect all embeddings
        all_audio_projected = []
        all_visual = []

        with torch.no_grad():
            for batch in self.test_loader:
                audio_emb = batch['audio_embedding'].to(self.device)
                visual_emb = batch['visual_embedding'].to(self.device)

                audio_projected = self.projection(audio_emb)

                all_audio_projected.append(audio_projected.cpu())
                all_visual.append(visual_emb.cpu())

        all_audio = torch.cat(all_audio_projected, dim=0)
        all_visual = torch.cat(all_visual, dim=0)

        # Normalize
        all_audio = F.normalize(all_audio, dim=1)
        all_visual = F.normalize(all_visual, dim=1)

        # Compute similarity matrix [N, N]
        sim = torch.matmul(all_audio, all_visual.T)

        # Find ranks
        n = sim.size(0)
        ranks = []
        top1_correct = 0
        top5_correct = 0

        for i in range(n):
            # Sort by similarity (descending)
            sorted_indices = sim[i].argsort(descending=True)

            # Find rank of correct answer (position of index i in sorted list)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

            if rank == 1:
                top1_correct += 1
            if rank <= 5:
                top5_correct += 1

        return {
            'top1_accuracy': top1_correct / n,
            'top5_accuracy': top5_correct / n,
            'mean_rank': sum(ranks) / n
        }

    def train(self):
        """Full training loop"""
        self.setup()

        print("\n" + "=" * 70)
        print("Starting Training")
        print("=" * 70)

        best_accuracy = 0.0

        for epoch in range(1, self.config.num_epochs + 1):
            # Train one epoch
            train_loss = self.train_epoch()

            # Log epoch
            epoch_log = {
                'epoch': epoch,
                'train_loss': train_loss,
                'timestamp': datetime.now().isoformat()
            }

            # Evaluate periodically
            if epoch % self.config.eval_every == 0 or epoch == 1:
                metrics = self.evaluate_retrieval()
                epoch_log.update(metrics)

                print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, "
                      f"Top1={metrics['top1_accuracy']:.3f}, "
                      f"Top5={metrics['top5_accuracy']:.3f}, "
                      f"MeanRank={metrics['mean_rank']:.1f}")

                # Save best model
                if metrics['top1_accuracy'] > best_accuracy:
                    best_accuracy = metrics['top1_accuracy']
                    self.history['best_epoch'] = epoch
                    self.history['best_top1_accuracy'] = best_accuracy
                    self._save_checkpoint('best')
            else:
                print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}")

            self.history['epochs'].append(epoch_log)

            # Periodic checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(f'epoch_{epoch}')

        # Save final model and history
        self._save_checkpoint('final')
        self._save_history()

        print("\n" + "=" * 70)
        print(f"Training complete!")
        print(f"Best Top-1 Accuracy: {best_accuracy:.3f} (epoch {self.history['best_epoch']})")
        print(f"Model saved to: {self.config.output_dir}")
        print("=" * 70)

    def _save_checkpoint(self, name: str):
        """Save projection head weights"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, f"projection_{name}.pt")
        torch.save(self.projection.state_dict(), path)

    def _save_history(self):
        """Save training history"""
        path = os.path.join(self.config.output_dir, "training_log.json")
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, name: str = 'best'):
        """Load projection head weights"""
        path = os.path.join(self.config.output_dir, f"projection_{name}.pt")
        self.projection.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded checkpoint: {path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Audio-Visual Contrastive Training")
    parser.add_argument("--data-dir", default="./data/vggsound", help="VGGSound data directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--checkpoint", default="best", help="Checkpoint to load for eval")
    args = parser.parse_args()

    config = AudioVisualTrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )

    trainer = AudioVisualTrainer(config)

    if args.eval_only:
        trainer.setup()
        trainer.load_checkpoint(args.checkpoint)
        metrics = trainer.evaluate_retrieval()
        print(f"Retrieval Metrics:")
        print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.3f}")
        print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.3f}")
        print(f"  Mean Rank: {metrics['mean_rank']:.1f}")
    else:
        trainer.train()


if __name__ == "__main__":
    main()
