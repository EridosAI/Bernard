"""
Audio-Visual Dataset for Contrastive Learning

PyTorch Dataset that loads paired audio-visual samples from VGGSound
and optionally precomputes embeddings for faster training.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import soundfile as sf
from PIL import Image
from tqdm import tqdm


@dataclass
class AudioVisualSample:
    """Metadata for a single audio-visual pair"""
    clip_id: str
    audio_path: str
    frame_path: str
    label: str
    label_idx: int


class VGGSoundDataset(Dataset):
    """
    Dataset of audio-visual pairs for contrastive learning.

    Can operate in two modes:
    1. Raw mode: Returns audio waveforms and frames (for on-the-fly encoding)
    2. Embedding mode: Returns precomputed embeddings (faster training)

    Args:
        data_dir: Path to VGGSound data directory
        beats_encoder: BEATs encoder for audio (required for embedding precomputation)
        vjepa_encoder: V-JEPA encoder for visual (required for embedding precomputation)
        precompute_embeddings: If True, compute and cache all embeddings
        split: 'train' or 'test' (80/20 split)
        transform: Optional transform for raw mode
    """

    def __init__(
        self,
        data_dir: str = "./data/vggsound",
        beats_encoder=None,
        vjepa_encoder=None,
        precompute_embeddings: bool = True,
        split: str = "train",
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.beats_encoder = beats_encoder
        self.vjepa_encoder = vjepa_encoder
        self.transform = transform
        self.split = split

        # Load samples from download log
        self.samples = self._load_samples()

        # Split into train/test
        self._apply_split()

        # Build label mapping
        self.labels = sorted(set(s.label for s in self.samples))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        # Update label indices in samples
        for sample in self.samples:
            sample.label_idx = self.label_to_idx[sample.label]

        # Embeddings cache
        self.embeddings_dir = self.data_dir / "embeddings"
        self.audio_embeddings: Optional[torch.Tensor] = None
        self.visual_embeddings: Optional[torch.Tensor] = None

        if precompute_embeddings:
            self._load_or_compute_embeddings()

    def _load_samples(self) -> List[AudioVisualSample]:
        """Load sample metadata from download log"""
        log_path = self.data_dir / "download_log.json"

        if not log_path.exists():
            raise FileNotFoundError(
                f"Download log not found at {log_path}\n"
                f"Run: python vggsound_downloader.py"
            )

        with open(log_path, 'r') as f:
            log = json.load(f)

        samples = []
        for clip_id, entry in log.items():
            if entry.get('status') != 'success':
                continue

            # Verify files exist
            audio_path = entry.get('audio_path', '')
            frame_path = entry.get('frame_path', '')

            if not os.path.exists(audio_path) or not os.path.exists(frame_path):
                continue

            samples.append(AudioVisualSample(
                clip_id=clip_id,
                audio_path=audio_path,
                frame_path=frame_path,
                label=entry['label'],
                label_idx=0  # Set later
            ))

        if not samples:
            raise ValueError(
                f"No valid samples found in {log_path}\n"
                f"Run: python vggsound_downloader.py"
            )

        return samples

    def _apply_split(self):
        """Apply train/test split (80/20, deterministic by clip_id)"""
        # Sort by clip_id for deterministic split
        all_samples = sorted(self.samples, key=lambda s: s.clip_id)

        # Use hash of clip_id for split to ensure same split every time
        train_samples = []
        test_samples = []

        for sample in all_samples:
            # Hash clip_id to get deterministic split
            h = hash(sample.clip_id) % 100
            if h < 80:
                train_samples.append(sample)
            else:
                test_samples.append(sample)

        if self.split == "train":
            self.samples = train_samples
        else:
            self.samples = test_samples

        print(f"Dataset split '{self.split}': {len(self.samples)} samples")

    def _load_or_compute_embeddings(self):
        """Load cached embeddings or compute them"""
        self.embeddings_dir.mkdir(exist_ok=True)

        cache_file = self.embeddings_dir / f"embeddings_{self.split}.pt"

        if cache_file.exists():
            print(f"Loading cached embeddings from {cache_file}...")
            cache = torch.load(cache_file, map_location='cpu')
            self.audio_embeddings = cache['audio']
            self.visual_embeddings = cache['visual']

            # Verify cache matches current samples
            if len(self.audio_embeddings) != len(self.samples):
                print("  Cache size mismatch, recomputing...")
                self._compute_embeddings()
                self._save_embeddings(cache_file)
        else:
            print("Computing embeddings (this may take a while)...")
            self._compute_embeddings()
            self._save_embeddings(cache_file)

    def _compute_embeddings(self):
        """Compute embeddings for all samples"""
        if self.beats_encoder is None or self.vjepa_encoder is None:
            raise ValueError(
                "beats_encoder and vjepa_encoder required for embedding computation.\n"
                "Either provide them or use precompute_embeddings=False."
            )

        audio_embs = []
        visual_embs = []

        for sample in tqdm(self.samples, desc="Computing embeddings"):
            # Audio embedding
            audio_emb = self.beats_encoder.encode_file(sample.audio_path)
            audio_embs.append(audio_emb.cpu())

            # Visual embedding
            visual_emb = self._encode_frame(sample.frame_path)
            visual_embs.append(visual_emb.cpu())

        self.audio_embeddings = torch.stack(audio_embs)
        self.visual_embeddings = torch.stack(visual_embs)

    def _encode_frame(self, frame_path: str) -> torch.Tensor:
        """Encode single frame with V-JEPA"""
        import cv2

        # Load frame
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # V-JEPA expects video (multiple frames), so we stack the same frame
        # This matches the pattern in bernard_integrated_v2.py
        frames = np.stack([frame] * 16)  # 16 frames minimum

        # Use VJEPAEncoder's encode_frames method
        return self.vjepa_encoder.encode_frames(frames)

    def _save_embeddings(self, cache_file: Path):
        """Save embeddings to cache"""
        torch.save({
            'audio': self.audio_embeddings,
            'visual': self.visual_embeddings,
            'clip_ids': [s.clip_id for s in self.samples],
            'labels': [s.label for s in self.samples]
        }, cache_file)
        print(f"  Saved embeddings to {cache_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get audio-visual pair.

        Returns:
            {
                'audio_embedding': Tensor[768],
                'visual_embedding': Tensor[1024],
                'label_idx': int
            }
        """
        sample = self.samples[idx]

        if self.audio_embeddings is not None:
            # Return precomputed embeddings
            return {
                'audio_embedding': self.audio_embeddings[idx],
                'visual_embedding': self.visual_embeddings[idx],
                'label_idx': sample.label_idx
            }
        else:
            # Compute on-the-fly (slower but more flexible)
            audio_emb = self.beats_encoder.encode_file(sample.audio_path)
            visual_emb = self._encode_frame(sample.frame_path)

            return {
                'audio_embedding': audio_emb,
                'visual_embedding': visual_emb,
                'label_idx': sample.label_idx
            }

    def get_label_name(self, label_idx: int) -> str:
        """Get label name from index"""
        return self.labels[label_idx]


class RawAudioVisualDataset(Dataset):
    """
    Dataset that returns raw audio waveforms and frames (no precomputed embeddings).

    Useful for debugging or when you want to encode on-the-fly with different models.
    """

    def __init__(
        self,
        data_dir: str = "./data/vggsound",
        split: str = "train",
        audio_sample_rate: int = 16000,
        frame_size: int = 256
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.audio_sample_rate = audio_sample_rate
        self.frame_size = frame_size

        # Load samples
        self.samples = self._load_samples()
        self._apply_split()

        # Build label mapping
        self.labels = sorted(set(s.label for s in self.samples))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        for sample in self.samples:
            sample.label_idx = self.label_to_idx[sample.label]

    def _load_samples(self) -> List[AudioVisualSample]:
        """Load sample metadata from download log"""
        log_path = self.data_dir / "download_log.json"

        with open(log_path, 'r') as f:
            log = json.load(f)

        samples = []
        for clip_id, entry in log.items():
            if entry.get('status') != 'success':
                continue

            audio_path = entry.get('audio_path', '')
            frame_path = entry.get('frame_path', '')

            if os.path.exists(audio_path) and os.path.exists(frame_path):
                samples.append(AudioVisualSample(
                    clip_id=clip_id,
                    audio_path=audio_path,
                    frame_path=frame_path,
                    label=entry['label'],
                    label_idx=0
                ))

        return samples

    def _apply_split(self):
        """Apply train/test split"""
        all_samples = sorted(self.samples, key=lambda s: s.clip_id)

        train_samples = []
        test_samples = []

        for sample in all_samples:
            h = hash(sample.clip_id) % 100
            if h < 80:
                train_samples.append(sample)
            else:
                test_samples.append(sample)

        self.samples = train_samples if self.split == "train" else test_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get raw audio-visual pair.

        Returns:
            {
                'audio': Tensor[samples],
                'frame': Tensor[3, H, W],
                'label_idx': int,
                'clip_id': str
            }
        """
        sample = self.samples[idx]

        # Load audio using soundfile directly (avoids torchaudio version issues)
        audio_data, sr = sf.read(sample.audio_path)
        waveform = torch.from_numpy(audio_data).float()

        # Handle stereo
        if waveform.dim() == 2:
            waveform = waveform.T.mean(dim=0)

        # Resample if needed
        if sr != self.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.audio_sample_rate)
            waveform = resampler(waveform)

        # Load frame
        frame = Image.open(sample.frame_path).convert('RGB')
        frame = frame.resize((self.frame_size, self.frame_size))
        frame = torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0

        return {
            'audio': waveform,
            'frame': frame,
            'label_idx': sample.label_idx,
            'clip_id': sample.clip_id
        }


class HowTo100MDataset(Dataset):
    """
    Dataset of audio-visual pairs from HowTo100M for contrastive learning.

    Similar to VGGSoundDataset but uses 'category' field instead of 'label'.
    HowTo100M contains instructional videos (DIY, crafts, cooking, etc.)

    Args:
        data_dir: Path to HowTo100M data directory
        beats_encoder: BEATs encoder for audio (required for embedding precomputation)
        vjepa_encoder: V-JEPA encoder for visual (required for embedding precomputation)
        precompute_embeddings: If True, compute and cache all embeddings
        split: 'train' or 'test' (80/20 split)
    """

    def __init__(
        self,
        data_dir: str = "./data/howto100m",
        beats_encoder=None,
        vjepa_encoder=None,
        precompute_embeddings: bool = True,
        split: str = "train"
    ):
        self.data_dir = Path(data_dir)
        self.beats_encoder = beats_encoder
        self.vjepa_encoder = vjepa_encoder
        self.split = split

        # Load samples from download log
        self.samples = self._load_samples()

        # Split into train/test
        self._apply_split()

        # Build label mapping (using category field)
        self.labels = sorted(set(s.label for s in self.samples))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        # Update label indices in samples
        for sample in self.samples:
            sample.label_idx = self.label_to_idx[sample.label]

        # Embeddings cache
        self.embeddings_dir = self.data_dir / "embeddings"
        self.audio_embeddings: Optional[torch.Tensor] = None
        self.visual_embeddings: Optional[torch.Tensor] = None

        if precompute_embeddings:
            self._load_or_compute_embeddings()

    def _load_samples(self) -> List[AudioVisualSample]:
        """Load sample metadata from download log"""
        log_path = self.data_dir / "download_log.json"

        if not log_path.exists():
            raise FileNotFoundError(
                f"Download log not found at {log_path}\n"
                f"Run: python howto100m_downloader.py"
            )

        with open(log_path, 'r') as f:
            log = json.load(f)

        samples = []
        for clip_id, entry in log.items():
            if entry.get('status') != 'success':
                continue

            # Verify files exist - handle both relative and absolute paths
            audio_path = entry.get('audio_path', '')
            frame_path = entry.get('frame_path', '')

            # Convert relative paths to absolute if needed
            if audio_path and not os.path.isabs(audio_path):
                audio_path = str(self.data_dir.parent.parent / audio_path)
            if frame_path and not os.path.isabs(frame_path):
                frame_path = str(self.data_dir.parent.parent / frame_path)

            if not os.path.exists(audio_path) or not os.path.exists(frame_path):
                continue

            # Use 'category' field for HowTo100M (e.g., "Hobbies and Crafts > Crafts")
            label = entry.get('category', 'unknown')

            samples.append(AudioVisualSample(
                clip_id=clip_id,
                audio_path=audio_path,
                frame_path=frame_path,
                label=label,
                label_idx=0  # Set later
            ))

        if not samples:
            raise ValueError(
                f"No valid samples found in {log_path}\n"
                f"Run: python howto100m_downloader.py"
            )

        return samples

    def _apply_split(self):
        """Apply train/test split (80/20, deterministic by clip_id)"""
        all_samples = sorted(self.samples, key=lambda s: s.clip_id)

        train_samples = []
        test_samples = []

        for sample in all_samples:
            h = hash(sample.clip_id) % 100
            if h < 80:
                train_samples.append(sample)
            else:
                test_samples.append(sample)

        if self.split == "train":
            self.samples = train_samples
        else:
            self.samples = test_samples

        print(f"HowTo100M split '{self.split}': {len(self.samples)} samples")

    def _load_or_compute_embeddings(self):
        """Load cached embeddings or compute them"""
        self.embeddings_dir.mkdir(exist_ok=True)

        cache_file = self.embeddings_dir / f"embeddings_{self.split}.pt"

        if cache_file.exists():
            print(f"Loading cached HowTo100M embeddings from {cache_file}...")
            cache = torch.load(cache_file, map_location='cpu')
            self.audio_embeddings = cache['audio']
            self.visual_embeddings = cache['visual']

            if len(self.audio_embeddings) != len(self.samples):
                print("  Cache size mismatch, recomputing...")
                self._compute_embeddings()
                self._save_embeddings(cache_file)
        else:
            print("Computing HowTo100M embeddings (this may take a while)...")
            self._compute_embeddings()
            self._save_embeddings(cache_file)

    def _compute_embeddings(self):
        """Compute embeddings for all samples"""
        if self.beats_encoder is None or self.vjepa_encoder is None:
            raise ValueError(
                "beats_encoder and vjepa_encoder required for embedding computation.\n"
                "Either provide them or use precompute_embeddings=False."
            )

        audio_embs = []
        visual_embs = []

        for sample in tqdm(self.samples, desc="Computing HowTo100M embeddings"):
            # Audio embedding
            audio_emb = self.beats_encoder.encode_file(sample.audio_path)
            audio_embs.append(audio_emb.cpu())

            # Visual embedding
            visual_emb = self._encode_frame(sample.frame_path)
            visual_embs.append(visual_emb.cpu())

        self.audio_embeddings = torch.stack(audio_embs)
        self.visual_embeddings = torch.stack(visual_embs)

    def _encode_frame(self, frame_path: str) -> torch.Tensor:
        """Encode single frame with V-JEPA"""
        import cv2

        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # V-JEPA expects video (multiple frames)
        frames = np.stack([frame] * 16)

        return self.vjepa_encoder.encode_frames(frames)

    def _save_embeddings(self, cache_file: Path):
        """Save embeddings to cache"""
        torch.save({
            'audio': self.audio_embeddings,
            'visual': self.visual_embeddings,
            'clip_ids': [s.clip_id for s in self.samples],
            'labels': [s.label for s in self.samples]
        }, cache_file)
        print(f"  Saved HowTo100M embeddings to {cache_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get audio-visual pair"""
        sample = self.samples[idx]

        if self.audio_embeddings is not None:
            return {
                'audio_embedding': self.audio_embeddings[idx],
                'visual_embedding': self.visual_embeddings[idx],
                'label_idx': sample.label_idx
            }
        else:
            audio_emb = self.beats_encoder.encode_file(sample.audio_path)
            visual_emb = self._encode_frame(sample.frame_path)

            return {
                'audio_embedding': audio_emb,
                'visual_embedding': visual_emb,
                'label_idx': sample.label_idx
            }

    def get_label_name(self, label_idx: int) -> str:
        """Get label name from index"""
        return self.labels[label_idx]


class CombinedAVDataset(Dataset):
    """
    Combined dataset that merges multiple audio-visual datasets.

    Concatenates samples from multiple datasets and creates a unified label space.
    Each source dataset maintains its own embedding cache.

    Args:
        datasets: List of (VGGSoundDataset, HowTo100MDataset, etc.)
    """

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.dataset_indices = []  # [(dataset_idx, sample_idx), ...]

        # Build unified sample index
        for ds_idx, ds in enumerate(self.datasets):
            for i in range(len(ds)):
                self.dataset_indices.append((ds_idx, i))

        # Build unified label mapping from all datasets
        all_labels = set()
        for ds in self.datasets:
            all_labels.update(ds.labels)
        self.labels = sorted(all_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        # Build reverse mapping: (dataset_idx, old_label_idx) -> new_label_idx
        self._label_remap = {}
        for ds_idx, ds in enumerate(self.datasets):
            for old_idx, label in enumerate(ds.labels):
                self._label_remap[(ds_idx, old_idx)] = self.label_to_idx[label]

        print(f"CombinedAVDataset: {len(self)} samples, {len(self.labels)} unique labels")
        for i, ds in enumerate(self.datasets):
            print(f"  Dataset {i}: {len(ds)} samples, {len(ds.labels)} labels")

    def __len__(self) -> int:
        return len(self.dataset_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get audio-visual pair with unified label index"""
        ds_idx, sample_idx = self.dataset_indices[idx]
        item = self.datasets[ds_idx][sample_idx]

        # Remap label_idx to unified space
        old_label_idx = item['label_idx']
        item['label_idx'] = self._label_remap[(ds_idx, old_label_idx)]

        return item

    def get_label_name(self, label_idx: int) -> str:
        """Get label name from unified index"""
        return self.labels[label_idx]


def test_dataset():
    """Test dataset loading"""
    print("Testing VGGSound dataset...")

    # Test raw dataset first (doesn't need encoders)
    try:
        raw_dataset = RawAudioVisualDataset(split="train")
        print(f"Raw dataset size: {len(raw_dataset)}")

        sample = raw_dataset[0]
        print(f"Audio shape: {sample['audio'].shape}")
        print(f"Frame shape: {sample['frame'].shape}")
        print(f"Label: {raw_dataset.labels[sample['label_idx']]}")
        print("Raw dataset working!")
    except FileNotFoundError as e:
        print(f"Dataset not downloaded yet: {e}")


def test_howto100m_dataset():
    """Test HowTo100M dataset loading"""
    print("\nTesting HowTo100M dataset...")

    try:
        dataset = HowTo100MDataset(precompute_embeddings=False, split="train")
        print(f"HowTo100M dataset size: {len(dataset)}")
        print(f"Number of categories: {len(dataset.labels)}")
        print(f"Sample categories: {dataset.labels[:5]}")
    except FileNotFoundError as e:
        print(f"Dataset not downloaded yet: {e}")


if __name__ == "__main__":
    test_dataset()
    test_howto100m_dataset()
