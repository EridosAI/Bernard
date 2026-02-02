"""
ImageBind Encoder Wrapper

Provides a unified interface to ImageBind's multimodal embedding space.
ImageBind maps audio, vision, and text to a shared 1024-dimensional space
where semantically similar concepts are close regardless of modality.

Used as the teacher model for distilling BEATs and V-JEPA embeddings
into a unified semantic space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from typing import Union, List, Optional
import numpy as np

# ImageBind imports
from imagebind import data as ib_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


class ImageBindEncoder:
    """
    Frozen ImageBind encoder for multimodal embeddings.

    All modalities (audio, vision, text) are projected to a shared
    1024-dimensional embedding space where cosine similarity
    indicates semantic relatedness.

    The model is kept frozen - this is used as a teacher for
    distillation, not for fine-tuning.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize ImageBind encoder.

        Args:
            device: Device to run on. Defaults to CUDA if available.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading ImageBind (imagebind_huge) on {device}...")
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model = self.model.to(device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        print("ImageBind loaded and frozen")

    def _load_audio_waveform(
        self,
        audio_path: str,
        target_sr: int = 16000,
        target_length: float = 2.0,
        num_clips: int = 3
    ) -> torch.Tensor:
        """
        Load audio file and prepare for ImageBind using soundfile.

        Uses soundfile instead of torchaudio to avoid torchcodec dependency.

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (ImageBind uses 16000)
            target_length: Length of each clip in seconds
            num_clips: Number of clips to sample

        Returns:
            Waveform tensor of shape [num_clips, samples]
        """
        import soundfile as sf

        # Load audio with soundfile
        audio_data, sr = sf.read(audio_path)
        waveform = torch.from_numpy(audio_data).float()

        # Convert to mono if stereo
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=1)

        # Resample if needed
        if sr != target_sr:
            # Simple linear interpolation resampling
            ratio = target_sr / sr
            new_length = int(len(waveform) * ratio)
            indices = torch.linspace(0, len(waveform) - 1, new_length)
            waveform = torch.from_numpy(
                np.interp(indices.numpy(), np.arange(len(waveform)), waveform.numpy())
            ).float()

        total_samples = waveform.shape[0]
        clip_samples = int(target_length * target_sr)

        # Sample clips uniformly
        clips = []
        if total_samples >= clip_samples:
            # Sample uniformly spaced clips
            max_start = total_samples - clip_samples
            starts = np.linspace(0, max_start, num_clips, dtype=int)
            for start in starts:
                clips.append(waveform[start:start + clip_samples])
        else:
            # Pad if audio is too short
            padded = F.pad(waveform, (0, clip_samples - total_samples))
            clips = [padded] * num_clips

        return torch.stack(clips)  # [num_clips, clip_samples]

    def _waveform_to_melspec(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        num_mel_bins: int = 128,
        target_length: int = 204
    ) -> torch.Tensor:
        """
        Convert waveform to mel spectrogram using ImageBind's expected format.

        Args:
            waveform: Audio waveform [num_clips, samples]
            sample_rate: Sample rate
            num_mel_bins: Number of mel frequency bins
            target_length: Target time dimension

        Returns:
            Mel spectrogram [num_clips, 1, num_mel_bins, target_length]
        """
        # ImageBind audio parameters
        mean = -4.268
        std = 9.138

        # Create mel spectrogram transform
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=num_mel_bins,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            mel_scale="slaney"
        )

        # Process each clip
        mel_specs = []
        for clip in waveform:
            # Compute mel spectrogram
            mel = mel_transform(clip)
            mel = torch.log(mel + 1e-10)

            # Normalize
            mel = (mel - mean) / std

            # Adjust time dimension to target_length
            if mel.shape[-1] > target_length:
                mel = mel[:, :target_length]
            elif mel.shape[-1] < target_length:
                mel = F.pad(mel, (0, target_length - mel.shape[-1]))

            mel_specs.append(mel.unsqueeze(0))  # [1, mel_bins, time]

        return torch.stack(mel_specs)  # [num_clips, 1, mel_bins, time]

    @torch.no_grad()
    def encode_audio(
        self,
        audio_paths: Union[str, Path, List[str], List[Path]]
    ) -> torch.Tensor:
        """
        Encode audio file(s) to embeddings.

        Uses custom audio loading to avoid torchcodec dependency.

        Args:
            audio_paths: Path or list of paths to audio files

        Returns:
            Normalized embeddings of shape [N, 1024] on CPU
        """
        if isinstance(audio_paths, (str, Path)):
            audio_paths = [str(audio_paths)]
        else:
            audio_paths = [str(p) for p in audio_paths]

        all_embeddings = []

        for audio_path in audio_paths:
            try:
                # Load and convert to mel spectrogram
                waveform = self._load_audio_waveform(audio_path)
                mel_spec = self._waveform_to_melspec(waveform)

                # Move to device
                mel_spec = mel_spec.to(self.device)

                # Forward pass through audio encoder
                inputs = {ModalityType.AUDIO: mel_spec}
                embeddings = self.model(inputs)

                # Get audio embedding and normalize
                audio_emb = embeddings[ModalityType.AUDIO]
                audio_emb = F.normalize(audio_emb, p=2, dim=-1)

                # Average over clips
                audio_emb = audio_emb.mean(dim=0, keepdim=True)
                all_embeddings.append(audio_emb)

            except Exception as e:
                print(f"Error encoding audio {audio_path}: {e}")
                # Return zero embedding on error
                all_embeddings.append(
                    torch.zeros(1, 1024, device=self.device)
                )

        return torch.cat(all_embeddings, dim=0).cpu()

    @torch.no_grad()
    def encode_vision(
        self,
        image_paths: Union[str, Path, List[str], List[Path]]
    ) -> torch.Tensor:
        """
        Encode image file(s) to embeddings.

        Args:
            image_paths: Path or list of paths to image files

        Returns:
            Normalized embeddings of shape [N, 1024] on CPU
        """
        if isinstance(image_paths, (str, Path)):
            image_paths = [str(image_paths)]
        else:
            image_paths = [str(p) for p in image_paths]

        # Load and transform vision data
        vision_data = ib_data.load_and_transform_vision_data(
            image_paths,
            self.device
        )

        # Forward pass
        inputs = {ModalityType.VISION: vision_data}
        embeddings = self.model(inputs)

        # Get vision embeddings and normalize
        vision_emb = embeddings[ModalityType.VISION]
        vision_emb = F.normalize(vision_emb, p=2, dim=-1)

        return vision_emb.cpu()

    @torch.no_grad()
    def encode_text(
        self,
        texts: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Encode text string(s) to embeddings.

        Args:
            texts: Text string or list of text strings

        Returns:
            Normalized embeddings of shape [N, 1024] on CPU
        """
        if isinstance(texts, str):
            texts = [texts]

        # Load and transform text data
        text_data = ib_data.load_and_transform_text(
            texts,
            self.device
        )

        # Forward pass
        inputs = {ModalityType.TEXT: text_data}
        embeddings = self.model(inputs)

        # Get text embeddings and normalize
        text_emb = embeddings[ModalityType.TEXT]
        text_emb = F.normalize(text_emb, p=2, dim=-1)

        return text_emb.cpu()

    @torch.no_grad()
    def encode_multimodal(
        self,
        audio_paths: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
        texts: Optional[List[str]] = None
    ) -> dict:
        """
        Encode multiple modalities in a single forward pass.

        This is more efficient than calling individual encode methods
        when you need embeddings for multiple modalities.

        Args:
            audio_paths: Optional list of audio file paths
            image_paths: Optional list of image file paths
            texts: Optional list of text strings

        Returns:
            Dictionary with keys 'audio', 'vision', 'text' mapping
            to normalized embeddings on CPU (only for provided modalities)
        """
        inputs = {}

        if audio_paths is not None:
            inputs[ModalityType.AUDIO] = ib_data.load_and_transform_audio_data(
                audio_paths, self.device
            )

        if image_paths is not None:
            inputs[ModalityType.VISION] = ib_data.load_and_transform_vision_data(
                image_paths, self.device
            )

        if texts is not None:
            inputs[ModalityType.TEXT] = ib_data.load_and_transform_text(
                texts, self.device
            )

        if not inputs:
            raise ValueError("At least one modality must be provided")

        # Forward pass
        embeddings = self.model(inputs)

        # Normalize and convert to CPU
        result = {}
        if ModalityType.AUDIO in embeddings:
            result['audio'] = F.normalize(
                embeddings[ModalityType.AUDIO], p=2, dim=-1
            ).cpu()
        if ModalityType.VISION in embeddings:
            result['vision'] = F.normalize(
                embeddings[ModalityType.VISION], p=2, dim=-1
            ).cpu()
        if ModalityType.TEXT in embeddings:
            result['text'] = F.normalize(
                embeddings[ModalityType.TEXT], p=2, dim=-1
            ).cpu()

        return result

    @staticmethod
    def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings.

        Args:
            emb1: Embeddings of shape [N, D] or [D]
            emb2: Embeddings of shape [M, D] or [D]

        Returns:
            Similarity scores. If both inputs are 1D, returns scalar.
            If one is 1D and other is 2D, returns [M] or [N].
            If both are 2D, returns [N, M] pairwise similarities.
        """
        # Handle 1D inputs
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)

        # Normalize (should already be normalized, but just in case)
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)

        # Compute similarity
        sim = emb1 @ emb2.T

        return sim.squeeze()


def run_sanity_test():
    """
    Run a sanity test to verify ImageBind is working correctly.

    Tests semantic alignment by comparing embeddings of related
    and unrelated concepts across modalities.
    """
    import tempfile
    import numpy as np
    from scipy.io import wavfile

    print("\n" + "="*60)
    print("ImageBind Sanity Test")
    print("="*60)

    # Initialize encoder
    encoder = ImageBindEncoder()

    # Test text encoding
    print("\n1. Testing text encoding...")
    concepts = ["a hammer", "a screwdriver", "a cat", "music playing"]
    text_embs = encoder.encode_text(concepts)
    print(f"   Text embeddings shape: {text_embs.shape}")

    # Compute pairwise text similarities
    print("\n   Text-Text similarities:")
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if i < j:
                sim = encoder.cosine_similarity(text_embs[i], text_embs[j])
                print(f"   '{c1}' <-> '{c2}': {sim.item():.3f}")

    # Create a simple test image (solid color)
    print("\n2. Testing vision encoding...")
    try:
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            # Create a simple test image
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            img.save(f.name)

            vision_emb = encoder.encode_vision(f.name)
            print(f"   Vision embedding shape: {vision_emb.shape}")

            # Compare to text concepts
            print("\n   Vision-Text similarities (gray image):")
            for i, concept in enumerate(concepts):
                sim = encoder.cosine_similarity(vision_emb[0], text_embs[i])
                print(f"   image <-> '{concept}': {sim.item():.3f}")

            # Clean up
            Path(f.name).unlink()
    except Exception as e:
        print(f"   Vision test skipped: {e}")

    # Test audio encoding with a generated tone
    print("\n3. Testing audio encoding...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Generate a simple sine wave
            sample_rate = 16000
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            # 440 Hz tone (A4)
            audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
            wavfile.write(f.name, sample_rate, audio)

            audio_emb = encoder.encode_audio(f.name)
            print(f"   Audio embedding shape: {audio_emb.shape}")

            # Compare to text concepts
            print("\n   Audio-Text similarities (440Hz tone):")
            for i, concept in enumerate(concepts):
                sim = encoder.cosine_similarity(audio_emb[0], text_embs[i])
                print(f"   tone <-> '{concept}': {sim.item():.3f}")

            # Clean up
            Path(f.name).unlink()
    except Exception as e:
        print(f"   Audio test skipped: {e}")

    print("\n" + "="*60)
    print("Sanity test complete!")
    print("="*60)

    return encoder


if __name__ == "__main__":
    run_sanity_test()
