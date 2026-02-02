"""
ImageBind Distillation Training

Trains projection heads to map BEATs (768-dim) and V-JEPA (1024-dim) embeddings
into ImageBind's shared semantic space (1024-dim). This allows Bernard to
inherit ImageBind's audio-visual-text alignment without the full model overhead.

Teacher (frozen): ImageBind huge
Students (frozen): BEATs, V-JEPA
Trainable: AudioProjection (768 -> 1024), VisualProjection (1024 -> 1024)

Loss: MSE between normalized embeddings (cosine alignment)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import cv2
import soundfile as sf


@dataclass
class DistillationConfig:
    """Configuration for distillation training"""
    # Data
    data_dir: str = "./data"
    vggsound_dir: str = "./data/vggsound"
    howto100m_dir: str = "./data/howto100m"

    # Model
    beats_dim: int = 768
    vjepa_dim: int = 1024
    imagebind_dim: int = 1024
    hidden_dim: int = 512

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    patience: int = 10  # Early stopping patience

    # Checkpoints
    checkpoint_dir: str = "./models/imagebind_distillation"
    save_every: int = 5


class AudioProjection(nn.Module):
    """
    Projects BEATs embeddings (768-dim) to ImageBind space (1024-dim).

    Architecture:
        Linear(768 -> 512) -> GELU -> Dropout ->
        Linear(512 -> 512) -> GELU -> Dropout ->
        Linear(512 -> 1024)

    Output is L2-normalized to match ImageBind's normalized embeddings.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: BEATs embeddings [B, 768]

        Returns:
            Normalized embeddings [B, 1024]
        """
        x = self.projection(x)
        return F.normalize(x, p=2, dim=-1)


class VisualProjection(nn.Module):
    """
    Projects V-JEPA embeddings (1024-dim) to ImageBind space (1024-dim).

    Even though dimensions match, this projection aligns the semantic spaces.
    V-JEPA's embedding space is video-focused; ImageBind's is multimodal.

    Architecture:
        Linear(1024 -> 512) -> GELU -> Dropout ->
        Linear(512 -> 512) -> GELU -> Dropout ->
        Linear(512 -> 1024)

    Output is L2-normalized to match ImageBind's normalized embeddings.
    """

    def __init__(
        self,
        input_dim: int = 1024,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: V-JEPA embeddings [B, 1024]

        Returns:
            Normalized embeddings [B, 1024]
        """
        x = self.projection(x)
        return F.normalize(x, p=2, dim=-1)


class DistillationDataset(Dataset):
    """
    Dataset for ImageBind distillation that returns embeddings from all encoders.

    For efficiency, embeddings can be precomputed and cached.

    Returns:
        {
            'beats_emb': BEATs embedding [768],
            'vjepa_emb': V-JEPA embedding [1024],
            'ib_audio_emb': ImageBind audio embedding [1024],
            'ib_visual_emb': ImageBind visual embedding [1024],
            'label': str
        }
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        cache_path: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.cache_path = Path(cache_path) if cache_path else None

        # These will be populated by precompute or loading
        self.beats_embs: Optional[torch.Tensor] = None
        self.vjepa_embs: Optional[torch.Tensor] = None
        self.ib_audio_embs: Optional[torch.Tensor] = None
        self.ib_visual_embs: Optional[torch.Tensor] = None
        self.labels: List[str] = []
        self.sample_ids: List[str] = []

        # Try to load from cache
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    def _load_cache(self):
        """Load precomputed embeddings from cache"""
        print(f"Loading distillation cache from {self.cache_path}...")
        cache = torch.load(self.cache_path, map_location='cpu')

        self.beats_embs = cache['beats_embs']
        self.vjepa_embs = cache['vjepa_embs']
        self.ib_audio_embs = cache['ib_audio_embs']
        self.ib_visual_embs = cache['ib_visual_embs']
        self.labels = cache['labels']
        self.sample_ids = cache['sample_ids']

        print(f"  Loaded {len(self.labels)} samples")

    def save_cache(self, cache_path: str):
        """Save computed embeddings to cache"""
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'beats_embs': self.beats_embs,
            'vjepa_embs': self.vjepa_embs,
            'ib_audio_embs': self.ib_audio_embs,
            'ib_visual_embs': self.ib_visual_embs,
            'labels': self.labels,
            'sample_ids': self.sample_ids
        }, cache_path)
        print(f"Saved distillation cache to {cache_path}")

    def __len__(self) -> int:
        return len(self.labels) if self.labels else 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'beats_emb': self.beats_embs[idx],
            'vjepa_emb': self.vjepa_embs[idx],
            'ib_audio_emb': self.ib_audio_embs[idx],
            'ib_visual_emb': self.ib_visual_embs[idx],
            'label': self.labels[idx]
        }


def precompute_embeddings(
    data_sources: List[str],
    output_dir: str,
    device: str = None
) -> Tuple[DistillationDataset, DistillationDataset]:
    """
    Precompute all embeddings for distillation training.

    Args:
        data_sources: List of data directories (vggsound, howto100m)
        output_dir: Where to save embedding caches
        device: Device to use for computation

    Returns:
        (train_dataset, val_dataset)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cache = output_dir / "distillation_train.pt"
    val_cache = output_dir / "distillation_val.pt"

    # Check if cache exists
    if train_cache.exists() and val_cache.exists():
        train_ds = DistillationDataset(".", split="train", cache_path=str(train_cache))
        val_ds = DistillationDataset(".", split="val", cache_path=str(val_cache))
        return train_ds, val_ds

    print("=" * 60)
    print("Precomputing embeddings for distillation")
    print("=" * 60)

    # Load encoders
    print("\nLoading encoders...")

    # BEATs encoder
    from beats_encoder import BEATsEncoder
    beats = BEATsEncoder()

    # V-JEPA encoder - use standalone version
    from vjepa_encoder import VJEPAEncoder

    # Find adapter if exists
    adapters_dir = Path("./models/adapters")
    adapter_path = None
    if adapters_dir.exists():
        adapters = list(adapters_dir.glob("workshop_lora_*"))
        if adapters:
            adapter_path = str(sorted(adapters)[-1])  # Most recent
            print(f"  Using adapter: {adapter_path}")

    vjepa = VJEPAEncoder(adapter_path=adapter_path)

    # ImageBind encoder
    from imagebind_encoder import ImageBindEncoder
    imagebind = ImageBindEncoder(device=device)

    # Collect all samples from data sources
    all_samples = []

    for source_dir in data_sources:
        source_path = Path(source_dir)
        log_file = source_path / "download_log.json"

        if not log_file.exists():
            print(f"  Skipping {source_dir} (no download_log.json)")
            continue

        print(f"  Loading samples from {source_dir}...")

        with open(log_file, 'r') as f:
            log = json.load(f)

        for clip_id, entry in log.items():
            if entry.get('status') != 'success':
                continue

            audio_path = entry.get('audio_path', '')
            frame_path = entry.get('frame_path', '')

            # Handle relative paths
            if audio_path and not os.path.isabs(audio_path):
                audio_path = str(source_path.parent.parent / audio_path)
            if frame_path and not os.path.isabs(frame_path):
                frame_path = str(source_path.parent.parent / frame_path)

            if os.path.exists(audio_path) and os.path.exists(frame_path):
                label = entry.get('label', entry.get('category', 'unknown'))
                all_samples.append({
                    'clip_id': clip_id,
                    'audio_path': audio_path,
                    'frame_path': frame_path,
                    'label': label
                })

    print(f"\nTotal samples found: {len(all_samples)}")

    if not all_samples:
        raise ValueError("No valid samples found. Run data downloaders first.")

    # Split into train/val (80/20)
    all_samples = sorted(all_samples, key=lambda x: x['clip_id'])
    train_samples = []
    val_samples = []

    for sample in all_samples:
        h = hash(sample['clip_id']) % 100
        if h < 80:
            train_samples.append(sample)
        else:
            val_samples.append(sample)

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    # Compute embeddings
    def compute_for_split(samples: List[dict], split_name: str) -> DistillationDataset:
        print(f"\n{split_name.upper()}: Computing embeddings for {len(samples)} samples...")

        beats_embs = []
        vjepa_embs = []
        ib_audio_embs = []
        ib_visual_embs = []
        labels = []
        sample_ids = []

        for sample in tqdm(samples, desc=f"Encoding {split_name}"):
            try:
                # BEATs embedding
                beats_emb = beats.encode_file(sample['audio_path'])
                beats_embs.append(beats_emb.cpu())

                # V-JEPA embedding
                frame = cv2.imread(sample['frame_path'])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames = np.stack([frame] * 16)  # V-JEPA needs video
                vjepa_emb = vjepa.encode_frames(frames)
                vjepa_embs.append(vjepa_emb.cpu())

                # ImageBind embeddings
                ib_audio = imagebind.encode_audio(sample['audio_path'])
                ib_audio_embs.append(ib_audio[0])  # Remove batch dim

                ib_visual = imagebind.encode_vision(sample['frame_path'])
                ib_visual_embs.append(ib_visual[0])  # Remove batch dim

                labels.append(sample['label'])
                sample_ids.append(sample['clip_id'])

            except Exception as e:
                print(f"  Error processing {sample['clip_id']}: {e}")
                continue

        # Create dataset
        ds = DistillationDataset(".", split=split_name)
        ds.beats_embs = torch.stack(beats_embs)
        ds.vjepa_embs = torch.stack(vjepa_embs)
        ds.ib_audio_embs = torch.stack(ib_audio_embs)
        ds.ib_visual_embs = torch.stack(ib_visual_embs)
        ds.labels = labels
        ds.sample_ids = sample_ids

        return ds

    # Compute for both splits
    train_ds = compute_for_split(train_samples, "train")
    val_ds = compute_for_split(val_samples, "val")

    # Save caches
    train_ds.save_cache(str(train_cache))
    val_ds.save_cache(str(val_cache))

    return train_ds, val_ds


class DistillationTrainer:
    """
    Trainer for ImageBind distillation.

    Trains AudioProjection and VisualProjection to align
    BEATs/V-JEPA embeddings with ImageBind's semantic space.
    """

    def __init__(self, config: DistillationConfig, device: str = None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Create projection heads
        self.audio_proj = AudioProjection(
            input_dim=config.beats_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.imagebind_dim
        ).to(self.device)

        self.visual_proj = VisualProjection(
            input_dim=config.vjepa_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.imagebind_dim
        ).to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            list(self.audio_proj.parameters()) + list(self.visual_proj.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Checkpoint dir
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.

        Loss = MSE(audio_proj(beats), ib_audio) + MSE(visual_proj(vjepa), ib_visual)

        Both are computed on normalized embeddings, which makes it equivalent
        to cosine similarity alignment.
        """
        # Move to device
        beats_emb = batch['beats_emb'].to(self.device)
        vjepa_emb = batch['vjepa_emb'].to(self.device)
        ib_audio = batch['ib_audio_emb'].to(self.device)
        ib_visual = batch['ib_visual_emb'].to(self.device)

        # Project student embeddings
        pred_audio = self.audio_proj(beats_emb)
        pred_visual = self.visual_proj(vjepa_emb)

        # Normalize targets (should already be normalized, but ensure)
        target_audio = F.normalize(ib_audio, p=2, dim=-1)
        target_visual = F.normalize(ib_visual, p=2, dim=-1)

        # MSE loss on normalized embeddings
        audio_loss = F.mse_loss(pred_audio, target_audio)
        visual_loss = F.mse_loss(pred_visual, target_visual)

        total_loss = audio_loss + visual_loss

        # Compute cosine similarity for monitoring (not loss)
        audio_cosine = (pred_audio * target_audio).sum(dim=-1).mean()
        visual_cosine = (pred_visual * target_visual).sum(dim=-1).mean()

        return {
            'total_loss': total_loss,
            'audio_loss': audio_loss,
            'visual_loss': visual_loss,
            'audio_cosine': audio_cosine,
            'visual_cosine': visual_cosine
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.audio_proj.train()
        self.visual_proj.train()

        total_loss = 0.0
        audio_loss = 0.0
        visual_loss = 0.0
        audio_cosine = 0.0
        visual_cosine = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {self.epoch}"):
            self.optimizer.zero_grad()

            losses = self.compute_loss(batch)
            losses['total_loss'].backward()
            self.optimizer.step()

            total_loss += losses['total_loss'].item()
            audio_loss += losses['audio_loss'].item()
            visual_loss += losses['visual_loss'].item()
            audio_cosine += losses['audio_cosine'].item()
            visual_cosine += losses['visual_cosine'].item()
            n_batches += 1

        return {
            'total_loss': total_loss / n_batches,
            'audio_loss': audio_loss / n_batches,
            'visual_loss': visual_loss / n_batches,
            'audio_cosine': audio_cosine / n_batches,
            'visual_cosine': visual_cosine / n_batches
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on held-out data"""
        self.audio_proj.eval()
        self.visual_proj.eval()

        total_loss = 0.0
        audio_loss = 0.0
        visual_loss = 0.0
        audio_cosine = 0.0
        visual_cosine = 0.0
        n_batches = 0

        for batch in val_loader:
            losses = self.compute_loss(batch)

            total_loss += losses['total_loss'].item()
            audio_loss += losses['audio_loss'].item()
            visual_loss += losses['visual_loss'].item()
            audio_cosine += losses['audio_cosine'].item()
            visual_cosine += losses['visual_cosine'].item()
            n_batches += 1

        return {
            'total_loss': total_loss / n_batches,
            'audio_loss': audio_loss / n_batches,
            'visual_loss': visual_loss / n_batches,
            'audio_cosine': audio_cosine / n_batches,
            'visual_cosine': visual_cosine / n_batches
        }

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'audio_proj_state': self.audio_proj.state_dict(),
            'visual_proj_state': self.visual_proj.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            print(f"  New best model saved!")

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.audio_proj.load_state_dict(checkpoint['audio_proj_state'])
        self.visual_proj.load_state_dict(checkpoint['visual_proj_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from epoch {self.epoch}")

    def train(
        self,
        train_dataset: DistillationDataset,
        val_dataset: DistillationDataset
    ):
        """Full training loop with early stopping"""

        print("\n" + "=" * 60)
        print("Starting ImageBind Distillation Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Max epochs: {self.config.num_epochs}")
        print(f"Early stopping patience: {self.config.patience}")
        print("=" * 60)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues on Windows
            pin_memory=True if self.device == "cuda" else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == "cuda" else False
        )

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch + 1

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Log
            print(f"\nEpoch {self.epoch}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(audio: {train_metrics['audio_loss']:.4f}, "
                  f"visual: {train_metrics['visual_loss']:.4f})")
            print(f"  Train Cosine: audio={train_metrics['audio_cosine']:.4f}, "
                  f"visual={train_metrics['visual_cosine']:.4f}")
            print(f"  Val Loss: {val_metrics['total_loss']:.4f} "
                  f"(audio: {val_metrics['audio_loss']:.4f}, "
                  f"visual: {val_metrics['visual_loss']:.4f})")
            print(f"  Val Cosine: audio={val_metrics['audio_cosine']:.4f}, "
                  f"visual={val_metrics['visual_cosine']:.4f}")

            # Check for improvement
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            if self.epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(f"checkpoint_epoch_{self.epoch}.pt", is_best)

            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered (patience={self.config.patience})")
                break

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {self.checkpoint_dir / 'best_checkpoint.pt'}")
        print("=" * 60)


def load_projection_heads(
    checkpoint_path: str,
    device: str = None
) -> Tuple[AudioProjection, VisualProjection]:
    """
    Load trained projection heads from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load to

    Returns:
        (audio_projection, visual_projection)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    # Create projection heads with saved config
    audio_proj = AudioProjection(
        input_dim=config.get('beats_dim', 768),
        hidden_dim=config.get('hidden_dim', 512),
        output_dim=config.get('imagebind_dim', 1024)
    ).to(device)

    visual_proj = VisualProjection(
        input_dim=config.get('vjepa_dim', 1024),
        hidden_dim=config.get('hidden_dim', 512),
        output_dim=config.get('imagebind_dim', 1024)
    ).to(device)

    # Load weights
    audio_proj.load_state_dict(checkpoint['audio_proj_state'])
    visual_proj.load_state_dict(checkpoint['visual_proj_state'])

    # Set to eval mode
    audio_proj.eval()
    visual_proj.eval()

    return audio_proj, visual_proj


def run_training(
    vggsound_dir: str = "./data/vggsound",
    howto100m_dir: str = "./data/howto100m",
    output_dir: str = "./models/imagebind_distillation",
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_epochs: int = 100,
    patience: int = 10,
    device: str = None
):
    """
    Main training entry point.

    Args:
        vggsound_dir: Path to VGGSound data
        howto100m_dir: Path to HowTo100M data
        output_dir: Where to save checkpoints
        batch_size: Training batch size
        learning_rate: Learning rate for AdamW
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Device to use (cuda/cpu)
    """
    # Collect data sources
    data_sources = []
    if Path(vggsound_dir).exists():
        data_sources.append(vggsound_dir)
    if Path(howto100m_dir).exists():
        data_sources.append(howto100m_dir)

    if not data_sources:
        raise ValueError(
            "No data sources found. Please download VGGSound or HowTo100M data:\n"
            "  python scripts/vggsound_downloader.py\n"
            "  python scripts/howto100m_downloader.py"
        )

    # Precompute embeddings
    train_ds, val_ds = precompute_embeddings(
        data_sources=data_sources,
        output_dir=output_dir,
        device=device
    )

    # Configure training
    config = DistillationConfig(
        vggsound_dir=vggsound_dir,
        howto100m_dir=howto100m_dir,
        checkpoint_dir=output_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        patience=patience
    )

    # Train
    trainer = DistillationTrainer(config, device=device)
    trainer.train(train_ds, val_ds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ImageBind Distillation Training")
    parser.add_argument("--vggsound-dir", type=str, default="./data/vggsound")
    parser.add_argument("--howto100m-dir", type=str, default="./data/howto100m")
    parser.add_argument("--output-dir", type=str, default="./models/imagebind_distillation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    run_training(
        vggsound_dir=args.vggsound_dir,
        howto100m_dir=args.howto100m_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        device=args.device
    )
