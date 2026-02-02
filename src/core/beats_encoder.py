"""
BEATs Audio Encoder

Wrapper for Microsoft BEATs (Bidirectional Encoder representation from Audio Transformers).
Produces 768-dimensional audio embeddings.

Setup:
    1. Clone https://github.com/microsoft/unilm
    2. Copy the unilm/beats directory to ./beats/
    3. Download BEATs_iter3_plus_AS2M.pt checkpoint to ./models/base/beats/
       URL: https://github.com/microsoft/unilm/blob/master/beats/README.md
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional
import numpy as np
import torch
import torchaudio
import soundfile as sf


class BEATsEncoder:
    """
    Wrapper for Microsoft BEATs audio encoder.

    Outputs 768-dimensional embeddings suitable for audio-visual alignment.
    Model stays frozen during training.
    """

    DEFAULT_CHECKPOINT = "./models/base/beats/BEATs_iter3_plus_AS2M.pt"
    BEATS_DIR = "./beats"  # Local copy of beats module

    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        checkpoint_path = checkpoint_path or self.DEFAULT_CHECKPOINT

        # Ensure beats module is importable
        # BEATs has internal relative imports (from backbone import ...) so we need
        # to add the beats directory itself to sys.path
        beats_path = Path(self.BEATS_DIR).resolve()
        if beats_path.exists():
            # Add beats dir for internal imports (backbone, modules, etc.)
            if str(beats_path) not in sys.path:
                sys.path.insert(0, str(beats_path))
            # Add parent for "from beats.X" imports
            if str(beats_path.parent) not in sys.path:
                sys.path.insert(0, str(beats_path.parent))

        # Import BEATs
        try:
            from BEATs import BEATs, BEATsConfig
        except ImportError as e:
            raise ImportError(
                f"Could not import BEATs. Please ensure the beats module is at {self.BEATS_DIR}\n"
                f"Setup: git clone https://github.com/microsoft/unilm && cp -r unilm/beats ./beats/\n"
                f"Original error: {e}"
            )

        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"BEATs checkpoint not found at {checkpoint_path}\n"
                f"Download from: https://github.com/microsoft/unilm/blob/master/beats/README.md"
            )

        print(f"Loading BEATs from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(device).eval()

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"  BEATs loaded ({self.embedding_dim}-dim output)")

    @property
    def embedding_dim(self) -> int:
        """Output embedding dimension"""
        return 768

    def encode(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000
    ) -> torch.Tensor:
        """
        Encode audio waveform to embedding.

        Args:
            audio: Waveform tensor/array. Shape [samples] or [batch, samples]
            sample_rate: Input sample rate (resampled to 16kHz if different)

        Returns:
            embedding: [768] for single audio, [batch, 768] for batch
        """
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Ensure float
        audio = audio.float()

        # Add batch dimension if needed
        single_input = audio.dim() == 1
        if single_input:
            audio = audio.unsqueeze(0)

        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio = resampler(audio)

        # Move to device
        audio = audio.to(self.device)

        # BEATs expects [batch, samples] and creates its own padding mask
        with torch.no_grad():
            # Create padding mask (all False = all valid)
            padding_mask = torch.zeros(audio.shape, dtype=torch.bool, device=self.device)

            # Extract features
            # BEATs returns (features, padding_mask) where features is [batch, time, 768]
            features, _ = self.model.extract_features(audio, padding_mask)

            # Mean pool over time dimension
            embedding = features.mean(dim=1)  # [batch, 768]

        if single_input:
            return embedding.squeeze(0)  # [768]
        return embedding  # [batch, 768]

    def encode_file(self, audio_path: str) -> torch.Tensor:
        """
        Load audio file and encode to embedding.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)

        Returns:
            embedding: [768] tensor
        """
        # Load audio using soundfile directly (avoids torchaudio version issues)
        audio_data, sample_rate = sf.read(audio_path)

        # Convert to torch tensor
        waveform = torch.from_numpy(audio_data).float()

        # Handle stereo: convert to shape [channels, samples]
        if waveform.dim() == 2:
            # Multi-channel: [samples, channels] -> [channels, samples]
            waveform = waveform.T
            # Convert stereo to mono
            waveform = waveform.mean(dim=0)
        # Now waveform is [samples]

        return self.encode(waveform, sample_rate)

    def encode_batch_files(self, audio_paths: list) -> torch.Tensor:
        """
        Encode multiple audio files as a batch.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            embeddings: [batch, 768] tensor
        """
        embeddings = []
        for path in audio_paths:
            emb = self.encode_file(path)
            embeddings.append(emb)
        return torch.stack(embeddings)


def setup_beats():
    """
    One-time setup for BEATs.

    Clones unilm repo and downloads checkpoint.
    """
    import subprocess

    beats_dir = Path("./beats")
    checkpoint_dir = Path("./models/base/beats")
    checkpoint_path = checkpoint_dir / "BEATs_iter3_plus_AS2M.pt"

    # Clone unilm and copy beats if needed
    if not beats_dir.exists():
        print("Setting up BEATs module...")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Sparse checkout just the beats directory
            subprocess.run([
                "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                "https://github.com/microsoft/unilm.git",
                tmpdir
            ], check=True)

            subprocess.run(
                ["git", "sparse-checkout", "set", "beats"],
                cwd=tmpdir, check=True
            )

            # Copy beats directory
            import shutil
            shutil.copytree(Path(tmpdir) / "beats", beats_dir)
            print(f"  Copied beats module to {beats_dir}")

    # Download checkpoint if needed
    if not checkpoint_path.exists():
        print("Downloading BEATs checkpoint...")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # The checkpoint URL from Microsoft's README
        checkpoint_url = "https://huggingface.co/liuronghuan/BEATs/resolve/main/BEATs_iter3_plus_AS2M.pt"

        subprocess.run([
            "curl", "-L", "-o", str(checkpoint_path), checkpoint_url
        ], check=True)
        print(f"  Downloaded to {checkpoint_path}")

    print("\nBEATs setup complete!")
    return True


def test_beats():
    """Quick test of BEATs encoder"""
    print("Testing BEATs encoder...")

    # Create test audio (1 second of noise)
    test_audio = np.random.randn(16000).astype(np.float32) * 0.1

    encoder = BEATsEncoder()
    embedding = encoder.encode(test_audio, sample_rate=16000)

    print(f"Input shape: {test_audio.shape}")
    print(f"Output shape: {embedding.shape}")
    print(f"Expected: ({encoder.embedding_dim},)")

    assert embedding.shape == (768,), f"Shape mismatch: {embedding.shape}"
    print("BEATs encoder working correctly!")

    return embedding


if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="BEATs Audio Encoder")
    parser.add_argument("--setup", action="store_true", help="Run setup (download model)")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--encode", type=str, help="Encode audio file")
    args = parser.parse_args()

    if args.setup:
        setup_beats()
    elif args.test:
        test_beats()
    elif args.encode:
        encoder = BEATsEncoder()
        embedding = encoder.encode_file(args.encode)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding (first 10): {embedding[:10].cpu().numpy()}")
    else:
        parser.print_help()
